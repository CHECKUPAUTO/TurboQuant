// TurboQuant Agent — Model Auditor & KV Cache Compression Advisor
//
// Scans Ollama models + GGUF files on disk, calculates TurboQuant
// compression projections, and reports savings.
//
// Modes:
//   turboquant-agent audit          → one-shot audit to stdout + JSON
//   turboquant-agent daemon         → periodic audit loop (systemd)
//   turboquant-agent watch          → continuous watch with delta detection
//
// 100% Rust — no Python.

use bytesize::ByteSize;
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;
use walkdir::WalkDir;

// ─────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
    pub modified_at: String,
    #[serde(default)]
    pub quantization_level: String,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub family: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufFile {
    pub path: String,
    pub filename: String,
    pub size_bytes: u64,
    pub size_human: String,
    pub quant_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionProjection {
    pub model_name: String,
    pub model_size_bytes: u64,
    pub estimated_kv_fp16_mb: f64,
    pub estimated_kv_turbo3_mb: f64,
    pub compression_ratio: f64,
    pub savings_mb: f64,
    pub ctx_sizes: Vec<CtxProjection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CtxProjection {
    pub ctx_size: usize,
    pub kv_fp16_mb: f64,
    pub kv_turbo3_mb: f64,
    pub savings_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NvmeInfo {
    pub device: String,
    pub mountpoint: String,
    pub total_bytes: u64,
    pub avail_bytes: u64,
    pub filesystem: String,
    pub is_nvme: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub timestamp: String,
    pub hostname: String,
    pub ollama_status: String,
    pub ollama_models: Vec<OllamaModel>,
    pub gguf_files: Vec<GgufFile>,
    pub projections: Vec<CompressionProjection>,
    pub nvme_disks: Vec<NvmeInfo>,
    pub total_models: usize,
    pub total_gguf_size_bytes: u64,
    pub total_kv_savings_mb: f64,
    pub vram_usage: Option<VramInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramInfo {
    pub total_mb: f64,
    pub used_mb: f64,
    pub free_mb: f64,
    pub gpu_name: String,
}

// ─────────────────────────────────────────────
// Ollama API client
// ─────────────────────────────────────────────

fn ollama_url() -> String {
    std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string())
}

/// List all Ollama models via /api/tags
fn list_ollama_models() -> Vec<OllamaModel> {
    let url = format!("{}/api/tags", ollama_url());

    let resp = match ureq::get(&url).timeout(Duration::from_secs(5)).call() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[WARN] Ollama non joignable: {e}");
            return vec![];
        }
    };

    let body: serde_json::Value = match resp.into_json() {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    let models = body["models"].as_array().cloned().unwrap_or_default();

    models
        .into_iter()
        .filter_map(|m| {
            Some(OllamaModel {
                name: m["name"].as_str()?.to_string(),
                model: m["model"].as_str().unwrap_or("").to_string(),
                size: m["size"].as_u64().unwrap_or(0),
                digest: m["digest"].as_str().unwrap_or("").to_string(),
                modified_at: m["modified_at"].as_str().unwrap_or("").to_string(),
                quantization_level: String::new(),
                parameter_size: String::new(),
                family: String::new(),
            })
        })
        .collect()
}

/// Get detailed model info via /api/show
fn get_model_details(name: &str) -> Option<(String, String, String)> {
    let url = format!("{}/api/show", ollama_url());

    let body = serde_json::json!({ "name": name });

    let resp = ureq::post(&url)
        .timeout(Duration::from_secs(10))
        .send_json(&body)
        .ok()?;

    let data: serde_json::Value = resp.into_json().ok()?;

    let details = &data["details"];
    let quant = details["quantization_level"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let params = details["parameter_size"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let family = details["family"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();

    Some((quant, params, family))
}

/// Enrich models with details from /api/show
fn enrich_models(models: &mut [OllamaModel]) {
    for m in models.iter_mut() {
        if let Some((quant, params, family)) = get_model_details(&m.name) {
            m.quantization_level = quant;
            m.parameter_size = params;
            m.family = family;
        }
    }
}

// ─────────────────────────────────────────────
// GGUF file scanner
// ─────────────────────────────────────────────

/// Detect quantization type from filename
fn detect_quant_from_name(name: &str) -> String {
    let upper = name.to_uppercase();
    let patterns = [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1",
        "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0", "F16", "F32", "IQ2_XXS", "IQ2_XS",
        "IQ3_XXS", "IQ3_XS", "IQ4_NL", "IQ4_XS",
    ];
    for p in &patterns {
        if upper.contains(p) {
            return p.to_string();
        }
    }
    "unknown".to_string()
}

/// Scan common directories for GGUF files
fn scan_gguf_files(extra_paths: &[&str]) -> Vec<GgufFile> {
    let mut search_dirs: Vec<PathBuf> = vec![
        PathBuf::from("/usr/share/ollama/.ollama/models"),
        PathBuf::from("/root/.ollama/models"),
        PathBuf::from("/home"),
        PathBuf::from("/opt/models"),
        PathBuf::from("/models"),
        PathBuf::from("/data/models"),
    ];

    // Add NVMe mountpoints
    if let Ok(mounts) = fs::read_to_string("/proc/mounts") {
        for line in mounts.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 && parts[0].contains("nvme") {
                search_dirs.push(PathBuf::from(parts[1]));
            }
        }
    }

    for p in extra_paths {
        search_dirs.push(PathBuf::from(p));
    }

    let mut found: Vec<GgufFile> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for dir in &search_dirs {
        if !dir.exists() {
            continue;
        }
        for entry in WalkDir::new(dir)
            .max_depth(6)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "gguf" {
                    let path_str = path.to_string_lossy().to_string();
                    if seen.contains(&path_str) {
                        continue;
                    }
                    seen.insert(path_str.clone());

                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    let filename = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    let quant = detect_quant_from_name(&filename);

                    found.push(GgufFile {
                        path: path_str,
                        filename,
                        size_bytes: size,
                        size_human: ByteSize(size).to_string(),
                        quant_type: quant,
                    });
                }
            }
        }
    }

    found.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    found
}

// ─────────────────────────────────────────────
// NVMe / Disk detection
// ─────────────────────────────────────────────

fn detect_nvme_disks() -> Vec<NvmeInfo> {
    let mut disks: Vec<NvmeInfo> = Vec::new();

    let output = Command::new("df")
        .args(["-B1", "--output=source,target,size,avail,fstype"])
        .output();

    let output = match output {
        Ok(o) => o,
        Err(_) => return disks,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines().skip(1) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }

        let device = parts[0];
        let mountpoint = parts[1];
        let total: u64 = parts[2].parse().unwrap_or(0);
        let avail: u64 = parts[3].parse().unwrap_or(0);
        let fstype = parts[4];

        let is_nvme = device.contains("nvme");
        let is_real = device.starts_with("/dev/");

        if is_real {
            disks.push(NvmeInfo {
                device: device.to_string(),
                mountpoint: mountpoint.to_string(),
                total_bytes: total,
                avail_bytes: avail,
                filesystem: fstype.to_string(),
                is_nvme,
            });
        }
    }

    // Sort: NVMe first, then by available space descending
    disks.sort_by(|a, b| {
        b.is_nvme
            .cmp(&a.is_nvme)
            .then(b.avail_bytes.cmp(&a.avail_bytes))
    });

    disks
}

/// Pick best install target: prefer NVMe with most space, fallback to largest disk
fn pick_install_target(disks: &[NvmeInfo]) -> Option<&NvmeInfo> {
    // Prefer NVMe
    let nvme: Vec<&NvmeInfo> = disks.iter().filter(|d| d.is_nvme).collect();
    if let Some(best) = nvme.into_iter().max_by_key(|d| d.avail_bytes) {
        return Some(best);
    }
    // Fallback: largest available
    disks.iter().max_by_key(|d| d.avail_bytes)
}

// ─────────────────────────────────────────────
// VRAM detection (nvidia-smi)
// ─────────────────────────────────────────────

fn detect_vram() -> Option<VramInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?;
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() < 4 {
        return None;
    }

    Some(VramInfo {
        gpu_name: parts[0].to_string(),
        total_mb: parts[1].parse().unwrap_or(0.0),
        used_mb: parts[2].parse().unwrap_or(0.0),
        free_mb: parts[3].parse().unwrap_or(0.0),
    })
}

// ─────────────────────────────────────────────
// TurboQuant compression projections
// ─────────────────────────────────────────────

/// Estimate model architecture params from parameter count string
fn estimate_arch(param_str: &str) -> (usize, usize, usize) {
    // (num_layers, num_heads, head_dim) — rough estimates
    let s = param_str.to_uppercase();

    if s.contains("70B") || s.contains("65B") {
        (80, 64, 128)
    } else if s.contains("34B") || s.contains("35B") {
        (60, 56, 128)
    } else if s.contains("27B") {
        (46, 32, 128)
    } else if s.contains("14B") || s.contains("13B") {
        (40, 40, 128)
    } else if s.contains("8B") || s.contains("7B") {
        (32, 32, 128)
    } else if s.contains("3B") || s.contains("4B") {
        (26, 32, 96)
    } else if s.contains("1B") || s.contains("2B") {
        (22, 16, 64)
    } else if s.contains("0.5B") || s.contains("500M") {
        (16, 16, 64)
    } else {
        // Default to 7B-class
        (32, 32, 128)
    }
}

/// Calculate KV cache size in MB for a given context size
fn kv_cache_mb(num_layers: usize, num_heads: usize, head_dim: usize, ctx: usize, bits: usize) -> f64 {
    // K + V = 2 × layers × heads × head_dim × ctx × bits/8
    let bytes = 2 * num_layers * num_heads * head_dim * ctx * bits / 8;
    bytes as f64 / (1024.0 * 1024.0)
}

fn compute_projections(models: &[OllamaModel]) -> Vec<CompressionProjection> {
    let ctx_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

    models
        .iter()
        .map(|m| {
            let (layers, heads, hdim) = estimate_arch(&m.parameter_size);

            let ctx_projs: Vec<CtxProjection> = ctx_sizes
                .iter()
                .map(|&ctx| {
                    let fp16 = kv_cache_mb(layers, heads, hdim, ctx, 16);
                    let turbo3 = kv_cache_mb(layers, heads, hdim, ctx, 3);
                    CtxProjection {
                        ctx_size: ctx,
                        kv_fp16_mb: fp16,
                        kv_turbo3_mb: turbo3,
                        savings_mb: fp16 - turbo3,
                    }
                })
                .collect();

            // Default projection at ctx=4096
            let default_fp16 = kv_cache_mb(layers, heads, hdim, 4096, 16);
            let default_turbo = kv_cache_mb(layers, heads, hdim, 4096, 3);

            CompressionProjection {
                model_name: m.name.clone(),
                model_size_bytes: m.size,
                estimated_kv_fp16_mb: default_fp16,
                estimated_kv_turbo3_mb: default_turbo,
                compression_ratio: if default_turbo > 0.0 {
                    default_fp16 / default_turbo
                } else {
                    0.0
                },
                savings_mb: default_fp16 - default_turbo,
                ctx_sizes: ctx_projs,
            }
        })
        .collect()
}

// ─────────────────────────────────────────────
// Report generation
// ─────────────────────────────────────────────

fn generate_report(extra_scan_paths: &[&str]) -> AuditReport {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let hostname = hostname();

    // Ollama models
    let mut models = list_ollama_models();
    let ollama_status = if models.is_empty() {
        "unreachable or no models".to_string()
    } else {
        format!("OK — {} models", models.len())
    };

    enrich_models(&mut models);

    // GGUF scan
    let gguf_files = scan_gguf_files(extra_scan_paths);
    let total_gguf: u64 = gguf_files.iter().map(|f| f.size_bytes).sum();

    // Projections
    let projections = compute_projections(&models);
    let total_savings: f64 = projections.iter().map(|p| p.savings_mb).sum();

    // Disks
    let nvme_disks = detect_nvme_disks();

    // VRAM
    let vram = detect_vram();

    AuditReport {
        timestamp,
        hostname,
        ollama_status,
        total_models: models.len(),
        ollama_models: models,
        gguf_files,
        projections,
        nvme_disks,
        total_gguf_size_bytes: total_gguf,
        total_kv_savings_mb: total_savings,
        vram_usage: vram,
    }
}

fn hostname() -> String {
    fs::read_to_string("/etc/hostname")
        .unwrap_or_else(|_| "unknown".to_string())
        .trim()
        .to_string()
}

// ─────────────────────────────────────────────
// Pretty-print report to terminal
// ─────────────────────────────────────────────

fn print_report(r: &AuditReport) {
    let sep = "═".repeat(70);
    let thin = "─".repeat(70);

    println!("\n{sep}");
    println!("  TURBOQUANT AGENT — AUDIT REPORT");
    println!("  {:<20} {}", "Timestamp:", r.timestamp);
    println!("  {:<20} {}", "Hostname:", r.hostname);
    println!("{sep}\n");

    // ── VRAM ──
    if let Some(ref vram) = r.vram_usage {
        println!("┌─ GPU ──────────────────────────────────────────┐");
        println!("│  {}",  vram.gpu_name);
        println!("│  VRAM: {:.0} / {:.0} MB (libre: {:.0} MB)", vram.used_mb, vram.total_mb, vram.free_mb);
        println!("└────────────────────────────────────────────────┘\n");
    }

    // ── NVMe / Disques ──
    println!("┌─ DISQUES ──────────────────────────────────────┐");
    for d in &r.nvme_disks {
        let tag = if d.is_nvme { " [NVMe]" } else { "" };
        println!(
            "│  {:<16} → {:<20} {:>10} dispo{}",
            d.device,
            d.mountpoint,
            ByteSize(d.avail_bytes).to_string(),
            tag
        );
    }
    println!("└────────────────────────────────────────────────┘\n");

    // ── Ollama ──
    println!("┌─ OLLAMA ({}) ─────────────────────────────────", r.ollama_status);
    println!("│");
    if r.ollama_models.is_empty() {
        println!("│  (aucun modèle détecté)");
    } else {
        println!(
            "│  {:<35} {:<10} {:<10} {:>10}",
            "Modèle", "Params", "Quant", "Taille"
        );
        println!("│  {thin}");
        for m in &r.ollama_models {
            println!(
                "│  {:<35} {:<10} {:<10} {:>10}",
                truncate(&m.name, 34),
                m.parameter_size,
                m.quantization_level,
                ByteSize(m.size).to_string()
            );
        }
    }
    println!("│");
    println!("└────────────────────────────────────────────────┘\n");

    // ── GGUF files ──
    if !r.gguf_files.is_empty() {
        println!("┌─ FICHIERS GGUF ({}) ──────────────────────────", r.gguf_files.len());
        println!("│");
        for f in r.gguf_files.iter().take(20) {
            println!(
                "│  {:<50} {:<10} {:>10}",
                truncate(&f.filename, 49),
                f.quant_type,
                f.size_human
            );
        }
        if r.gguf_files.len() > 20 {
            println!("│  ... +{} fichiers", r.gguf_files.len() - 20);
        }
        println!(
            "│\n│  Total GGUF: {}",
            ByteSize(r.total_gguf_size_bytes).to_string()
        );
        println!("└────────────────────────────────────────────────┘\n");
    }

    // ── Compression projections ──
    println!("┌─ PROJECTIONS TURBOQUANT (3-bit KV cache) ─────┐");
    println!("│");
    println!(
        "│  {:<30} {:>10} {:>10} {:>10} {:>8}",
        "Modèle", "KV FP16", "KV Turbo3", "Économie", "Ratio"
    );
    println!("│  {thin}");
    for p in &r.projections {
        println!(
            "│  {:<30} {:>8.1} MB {:>8.1} MB {:>8.1} MB {:>6.1}x",
            truncate(&p.model_name, 29),
            p.estimated_kv_fp16_mb,
            p.estimated_kv_turbo3_mb,
            p.savings_mb,
            p.compression_ratio
        );
    }
    println!("│");
    println!(
        "│  TOTAL ÉCONOMIE KV CACHE: {:.1} MB (ctx=4096)",
        r.total_kv_savings_mb
    );
    println!("└────────────────────────────────────────────────┘\n");

    // ── Extended ctx projection for first model ──
    if let Some(p) = r.projections.first() {
        println!("┌─ DÉTAIL: {} ─────────────", truncate(&p.model_name, 30));
        println!("│  {:<10} {:>10} {:>10} {:>10}", "Contexte", "KV FP16", "KV Turbo3", "Économie");
        for c in &p.ctx_sizes {
            println!(
                "│  {:<10} {:>8.1} MB {:>8.1} MB {:>8.1} MB",
                format!("{}k", c.ctx_size / 1024),
                c.kv_fp16_mb,
                c.kv_turbo3_mb,
                c.savings_mb
            );
        }
        println!("└────────────────────────────────────────────────┘\n");
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

// ─────────────────────────────────────────────
// Report persistence
// ─────────────────────────────────────────────

fn save_report(r: &AuditReport, dir: &Path) {
    let _ = fs::create_dir_all(dir);

    // Latest JSON (overwrite)
    let latest = dir.join("latest.json");
    if let Ok(json) = serde_json::to_string_pretty(r) {
        let _ = fs::write(&latest, &json);
    }

    // Timestamped JSON
    let ts = Local::now().format("%Y%m%d_%H%M%S");
    let stamped = dir.join(format!("audit_{ts}.json"));
    if let Ok(json) = serde_json::to_string_pretty(r) {
        let _ = fs::write(&stamped, &json);
    }

    // Rotate: keep last 100
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().starts_with("audit_"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();
    while entries.len() > 100 {
        if let Some(old) = entries.first() {
            let _ = fs::remove_file(old);
        }
        entries.remove(0);
    }
}

// ─────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────

fn print_usage() {
    eprintln!(
        r#"
TurboQuant Agent v1.0.0 — Audit modèles & projection compression KV cache

USAGE:
    turboquant-agent audit   [--json]     Audit one-shot
    turboquant-agent daemon  [--interval SECS]  Daemon systemd (défaut: 300s)
    turboquant-agent watch                Watch continu (delta)
    turboquant-agent disks                Lister disques NVMe
    turboquant-agent install-target       Afficher le meilleur disque cible
    turboquant-agent help                 Aide

VARIABLES D'ENV:
    OLLAMA_HOST       URL Ollama (défaut: http://127.0.0.1:11434)
    TQ_REPORT_DIR     Répertoire rapports (défaut: /var/log/turboquant)
    TQ_SCAN_PATHS     Chemins additionnels (séparés par :)
"#
    );
}

fn report_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("TQ_REPORT_DIR").unwrap_or_else(|_| "/var/log/turboquant".to_string()),
    )
}

fn extra_scan_paths() -> Vec<String> {
    std::env::var("TQ_SCAN_PATHS")
        .unwrap_or_default()
        .split(':')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "audit" => {
            let json_mode = args.iter().any(|a| a == "--json");
            let extra: Vec<String> = extra_scan_paths();
            let extra_refs: Vec<&str> = extra.iter().map(|s| s.as_str()).collect();

            let report = generate_report(&extra_refs);

            if json_mode {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                print_report(&report);
            }

            save_report(&report, &report_dir());
            eprintln!(
                "[OK] Rapport sauvé → {}",
                report_dir().join("latest.json").display()
            );
        }

        "daemon" => {
            let interval: u64 = args
                .iter()
                .position(|a| a == "--interval")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(300);

            eprintln!(
                "[DAEMON] TurboQuant Agent démarré — intervalle: {}s, rapports: {}",
                interval,
                report_dir().display()
            );

            loop {
                let extra: Vec<String> = extra_scan_paths();
                let extra_refs: Vec<&str> = extra.iter().map(|s| s.as_str()).collect();

                let report = generate_report(&extra_refs);
                print_report(&report);
                save_report(&report, &report_dir());

                eprintln!(
                    "[DAEMON] Audit terminé — {} modèles, économie totale: {:.1} MB — prochain dans {}s",
                    report.total_models, report.total_kv_savings_mb, interval
                );

                thread::sleep(Duration::from_secs(interval));
            }
        }

        "watch" => {
            eprintln!("[WATCH] Mode surveillance continue — Ctrl+C pour arrêter");
            let mut last_count = 0usize;
            let mut last_gguf = 0usize;

            loop {
                let extra: Vec<String> = extra_scan_paths();
                let extra_refs: Vec<&str> = extra.iter().map(|s| s.as_str()).collect();

                let report = generate_report(&extra_refs);
                let new_count = report.total_models;
                let new_gguf = report.gguf_files.len();

                if new_count != last_count || new_gguf != last_gguf {
                    println!("\n[DELTA] Changement détecté: {} → {} modèles, {} → {} GGUF",
                        last_count, new_count, last_gguf, new_gguf);
                    print_report(&report);
                    save_report(&report, &report_dir());
                    last_count = new_count;
                    last_gguf = new_gguf;
                } else {
                    eprint!(".");
                    std::io::stderr().flush().ok();
                }

                thread::sleep(Duration::from_secs(30));
            }
        }

        "disks" => {
            let disks = detect_nvme_disks();
            for d in &disks {
                let tag = if d.is_nvme { " ★ NVMe" } else { "" };
                println!(
                    "{:<16} {:<20} {:>12} dispo / {:>12} total  [{}]{}",
                    d.device,
                    d.mountpoint,
                    ByteSize(d.avail_bytes).to_string(),
                    ByteSize(d.total_bytes).to_string(),
                    d.filesystem,
                    tag
                );
            }
        }

        "install-target" => {
            let disks = detect_nvme_disks();
            match pick_install_target(&disks) {
                Some(d) => {
                    let tag = if d.is_nvme { "NVMe" } else { "HDD/SSD" };
                    println!("{}|{}|{}|{}", d.mountpoint, d.device, tag, d.avail_bytes);
                }
                None => {
                    eprintln!("[ERR] Aucun disque détecté");
                    std::process::exit(1);
                }
            }
        }

        _ => print_usage(),
    }
}
