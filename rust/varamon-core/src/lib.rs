//! Receive-side building blocks for VARA HF frames.
//!
//! This crate is the Rust port of the Python reference in this repository. Each module is checked
//! against values produced by that reference.

pub mod audio;
pub mod b2f;
pub mod crc;
pub mod ctrl;
pub mod frame;
pub mod fsk;
pub mod lzhuf;
pub mod scan;
pub mod tables;
pub mod textcomp;
pub mod turbo;
pub mod vb;
