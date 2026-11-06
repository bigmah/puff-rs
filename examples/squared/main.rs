//! A whole program on pufferl: one env, Squared (squared.rs), and the command line.
//!
//!     cargo run --release --example squared -- train squared
//!     cargo run --release --example squared -- eval squared latest
//!
//! An env of your own is the same three files: the env, its ini, and a main that adds it
//! to an Envs by name. More envs are more .add lines.

mod squared;

use pufferl::{cli, Envs};

fn main() {
    cli::main(Envs::new().add("squared", include_str!("squared.ini"), squared::Squared::new));
}
