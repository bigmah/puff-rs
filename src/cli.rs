//! The command line, for a program's envs:
//!
//! ```text
//! PROGRAM train ENV [--config FILE ...] [--section.key=value ...]
//! PROGRAM eval ENV [latest | MODEL.bin] [--headless] [--config FILE ...] [--section.key=value ...]
//! ```
//!
//! Checkpoints and logs go where base.checkpoint_dir and base.log_dir say, from the directory
//! it runs in. An [`Envs`] fallback looks for envs in that directory, or in PUFFERL_ROOT.

use std::path::PathBuf;
use std::rc::Rc;

use crate::config::load_config;
use crate::envs::Envs;
use crate::gpu::Gpu;
use crate::pufferl::{run_eval, run_train};

/// Where an [`Envs`] fallback looks for envs: PUFFERL_ROOT, or the directory this runs in
pub fn root() -> PathBuf {
    match std::env::var("PUFFERL_ROOT") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => std::env::current_dir().unwrap_or_default(),
    }
}

/// Aggressively exit on ctrl+c, as puffer does
#[cfg(unix)]
fn exit_on_interrupt() {
    extern "C" fn interrupted(_: libc::c_int) {
        unsafe { libc::_exit(0) }
    }
    unsafe {
        libc::signal(libc::SIGINT, interrupted as *const () as libc::sighandler_t);
    }
}

#[cfg(not(unix))]
fn exit_on_interrupt() {}

/// Train or eval one of envs, as the command line says
pub fn main(envs: Envs) {
    exit_on_interrupt();
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "pufferl".into());
    let args: Vec<String> = args.collect();
    let usage = format!("usage: {program} train|eval ENV [latest|MODEL.bin] [--headless] [--config FILE ...] [--section.key=value ...]");
    if args.len() < 2 || !matches!(args[0].as_str(), "train" | "eval") {
        let names: Vec<&str> = envs.names().collect();
        let listed = if names.is_empty() { String::new() } else { format!("\nenvs: {}", names.join(", ")) };
        crate::fail(&format!("{usage}{listed}"));
    }
    let (mode, env_name) = (args[0].as_str(), args[1].as_str());
    let mut rest: Vec<String> = args[2..].to_vec();
    let mut model = None;
    if mode == "eval" && rest.first().is_some_and(|arg| !arg.starts_with('-') && !arg.contains('=')) {
        model = Some(rest.remove(0));
    }
    let headless = rest.iter().any(|arg| arg == "--headless");
    rest.retain(|arg| arg != "--headless");

    let root = root();
    let Some(env) = envs.find(&root, env_name) else {
        let names: Vec<&str> = envs.names().collect();
        crate::fail(&format!("no env called {env_name}{}", if names.is_empty() { String::new() } else { format!(" (envs: {})", names.join(", ")) }));
    };
    let mut ini = load_config(env_name, env.config(), env.origin(), &rest);
    if let Some(model) = model {
        ini.put("base.load_model_path", &model);
    }
    let gpu = Rc::new(Gpu::new().unwrap_or_else(|e| crate::fail(&e)));
    match mode {
        "train" => run_train(&env, gpu, ini),
        _ => run_eval(&env, gpu, ini, !headless),
    }
}
