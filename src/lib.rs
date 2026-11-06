//! PufferLib 5.0's trainer on wgpu, for whatever GPU a machine has.
//!
//! PufferLib 5.0 trains with one CUDA program (src/pufferl.cu, src/algo.cu). This is that
//! trainer on wgpu, which is Metal on a Mac, Vulkan on Linux and DX12 on Windows: the same
//! policy (bias-free Linear encoder -> MinGRU -> one decoder matrix for logits and value),
//! the same PPO (advantages from each minibatch's own values, no prioritized replay,
//! recurrent state carried across horizons), Muon, the same config files and CLI overrides,
//! checkpoints PufferLib's own eval can read, and its dashboard. Like the CUDA trainer it
//! has no autograd: every backward pass is a kernel written for it (src/kernels).
//!
//! An env implements [`env::Env`] (the puffer-env crate, re-exported here). A program adds
//! its envs to an [`Envs`] by name and hands that to [`cli::main`], which trains or evals the
//! one the command line names, stepping it in pufferl's own vecenv (src/vecenv.rs):
//!
//! ```ignore
//! use pufferl::{cli, Envs};
//!
//! fn main() {
//!     cli::main(Envs::new().add("squared", include_str!("squared.ini"), Squared::new));
//! }
//! ```
//!
//! examples/squared is a whole one.

pub mod actor;
pub mod checkpoint;
pub mod cli;
pub mod config;
pub mod dashboard;
pub mod envs;
pub mod gpu;
pub mod learner;
pub mod model;
pub mod ops;
pub mod pufferl;
pub mod rng;
pub mod vecenv;

/// What an env is: the Env trait, its Config, Log and Agents
pub use puffer_env as env;
pub use envs::{Entry, Envs};
pub use vecenv::VecEnv;

/// Say what is wrong and stop, as sys.exit does with a message.
pub fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
