//! Receive-side building blocks for VARA HF frames.
//!
//! This crate is the Rust port of the Python reference in this repository. Each module is checked
//! against values produced by that reference.

pub mod crc;
pub mod tables;
pub mod turbo;
pub mod vb;
