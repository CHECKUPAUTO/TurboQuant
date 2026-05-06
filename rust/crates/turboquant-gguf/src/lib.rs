#![deny(missing_docs)]

//! GGUF I/O for `TurboQuant`.
//!
//! Handles parsing and writing of GGUF files with support for
//! the `cache-type turbo3` extension.

/// Parser for GGUF format files.
pub mod parser;
/// GGUF types and constants.
pub mod types;
/// Writer for GGUF format files.
pub mod writer;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
