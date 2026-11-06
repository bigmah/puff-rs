//! The Rust envs, by name. An env in envs/<name>/ with a Cargo.toml is a Rust env: it
//! implements puffer_env::Env, is a dependency of this crate, and has a line here saying
//! how to make one. Any other name is a C env, which build.sh builds (src/cenv.rs).

use puffer_env::Config;

use crate::vecenv::VecEnv;

/// The Rust env called name in a vecenv, or None if there is no Rust env by that name
pub fn native(name: &str, config: &Config, total_agents: usize, num_threads: usize) -> Option<VecEnv> {
    Some(match name {
        "squared" => VecEnv::from_envs(|i| squared::Squared::new(config, i), total_agents, num_threads),
        _ => return None,
    })
}
