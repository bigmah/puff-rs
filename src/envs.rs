//! The envs a program trains, by name. pufferl has none of its own: a program makes an
//! [`Envs`], adds its envs to it, and hands it to [`crate::cli::main`].
//!
//! ```ignore
//! fn main() {
//!     pufferl::cli::main(Envs::new().add("squared", include_str!("squared.ini"), Squared::new));
//! }
//! ```

use std::path::Path;
use std::rc::Rc;

use puffer_env::{Config, Env};

use crate::vecenv::VecEnv;

type MakeVec = dyn Fn(&Config, usize, usize) -> VecEnv;
type Fallback = dyn Fn(&Path, &str) -> Option<Entry>;

/// One env: its config, and how to make a vecenv of it
pub struct Entry {
    config: String,
    origin: String,
    make: Box<MakeVec>,
}

impl Entry {
    /// config is the text of the env's ini, which origin names in errors. make is given the
    /// [env] section, vec.total_agents and vec.num_threads.
    pub fn new(config: impl Into<String>, origin: impl Into<String>, make: impl Fn(&Config, usize, usize) -> VecEnv + 'static) -> Self {
        Self { config: config.into(), origin: origin.into(), make: Box::new(make) }
    }

    pub fn config(&self) -> &str {
        &self.config
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    pub fn make(&self, config: &Config, total_agents: usize, num_threads: usize) -> VecEnv {
        (self.make)(config, total_agents, num_threads)
    }
}

#[derive(Default)]
pub struct Envs {
    entries: Vec<(String, Rc<Entry>)>,
    fallback: Option<Box<Fallback>>,
}

impl Envs {
    pub fn new() -> Self {
        Self::default()
    }

    /// A Rust env called name. config is the text of its ini (include_str! of it, say), and
    /// make(config, i) makes the i-th of the envs a vecenv steps.
    pub fn add<E: Env + 'static>(self, name: &str, config: &str, make: impl Fn(&Config, usize) -> E + 'static) -> Self {
        let make = Rc::new(make);
        let entry = Entry::new(config, format!("{name}.ini"), move |env_config: &Config, total_agents, num_threads| {
            let make = make.clone();
            let env_config = env_config.clone();
            VecEnv::from_envs(move |i| make(&env_config, i), total_agents, num_threads)
        });
        self.add_entry(name, entry)
    }

    /// An env of any kind, a C env behind a bridge say, by its [`Entry`]
    pub fn add_entry(mut self, name: &str, entry: Entry) -> Self {
        self.entries.retain(|(n, _)| n != name);
        self.entries.push((name.to_string(), Rc::new(entry)));
        self
    }

    /// Where an env not added by name is looked for: given the directory the program runs
    /// in and the name, the env, or None if there is none by that name.
    pub fn fallback(mut self, find: impl Fn(&Path, &str) -> Option<Entry> + 'static) -> Self {
        self.fallback = Some(Box::new(find));
        self
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(name, _)| name.as_str())
    }

    /// The env called name, added by name or found by the fallback
    pub fn find(&self, root: &Path, name: &str) -> Option<Rc<Entry>> {
        if let Some((_, entry)) = self.entries.iter().find(|(n, _)| n == name) {
            return Some(entry.clone());
        }
        self.fallback.as_ref().and_then(|find| find(root, name)).map(Rc::new)
    }
}
