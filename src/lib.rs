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
//! Envs step in its own vecenv (src/vecenv.rs). An env is a Rust env, which implements
//! puffer_env::Env (crates/puffer-env) and is named in src/envs.rs, or a PufferLib 5.0 C
//! env, which build.sh compiles into a bridge library (cenv.c) that this loads:
//!
//! ```text
//! pufferl train platformer [--config FILE ...] [--section.key=value ...]
//! pufferl eval platformer [latest | MODEL.bin] [--headless] [--config FILE ...] [--section.key=value ...]
//! ```

pub mod actor;
pub mod cenv;
pub mod checkpoint;
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

/// Say what is wrong and stop, as sys.exit does with a message.
pub fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
