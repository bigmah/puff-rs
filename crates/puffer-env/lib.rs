//! What an env is to pufferl. A Rust env implements [`Env`] and is trained with no C in
//! between: the trainer makes envs until it has `vec.total_agents` agents, steps them
//! together on `vec.num_threads` threads, and hands each its own agents' slices of the
//! observation, action, reward and terminal buffers it trains on.
//!
//! ```ignore
//! impl Env for Squared {
//!     type Obs = u8;
//!     fn obs_size(&self) -> usize { self.size * self.size }
//!     fn act_sizes(&self) -> Vec<usize> { vec![5] }
//!     fn reset(&mut self, mut agents: Agents<u8>) { ... }
//!     fn step(&mut self, mut agents: Agents<u8>) { ... }
//!     fn log(&mut self) -> &mut Log { &mut self.log }
//! }
//! ```
//!
//! An env is registered with the trainer by name, in src/envs.rs, which says how to make
//! one from its [`Config`] and its index.
//!
//! This crate has no dependencies, so an env that uses it builds without the trainer.

/// The type of one element of an observation. The policy reads observations as floats.
pub trait Obs: Copy + Default + Send + Sync + 'static {
    fn to_f32(self) -> f32;
}

impl Obs for u8 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Obs for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
}

/// An env's agents' slices of the trainer's buffers, for one reset or step. Each holds
/// `num_agents` agents end to end: agent `i`'s observation is `obs_size` elements from
/// `i * obs_size`, and its actions are one float per action head from `i * num_atns`.
///
/// Rewards and terminals are zero when a step starts. Observations are as the env left
/// them, so an env that changes a few cells of its observation need only change those.
pub struct Agents<'a, O> {
    pub observations: &'a mut [O],
    pub actions: &'a [f32],
    pub rewards: &'a mut [f32],
    pub terminals: &'a mut [f32],
    /// One byte per action of each head, 1 where the action may be taken. All 1 to begin
    /// with; the trainer does not read it yet.
    pub action_mask: &'a mut [u8],
    pub obs_size: usize,
    pub num_atns: usize,
    pub mask_size: usize,
}

impl<'a, O> Agents<'a, O> {
    pub fn len(&self) -> usize {
        self.rewards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rewards.is_empty()
    }

    /// Agent `i`'s observation
    pub fn observation(&mut self, i: usize) -> &mut [O] {
        &mut self.observations[i * self.obs_size..(i + 1) * self.obs_size]
    }

    /// The action agent `i` took with head `head`
    pub fn action(&self, i: usize, head: usize) -> usize {
        self.actions[i * self.num_atns + head] as usize
    }

    /// Agent `i`'s mask, every head's actions end to end
    pub fn mask(&mut self, i: usize) -> &mut [u8] {
        &mut self.action_mask[i * self.mask_size..(i + 1) * self.mask_size]
    }
}

/// An env: one or more agents that reset and step together.
///
/// Every env the trainer makes must agree on `obs_size` and `act_sizes`, which are read
/// from the first; `num_agents` may differ from env to env.
pub trait Env: Send {
    type Obs: Obs;

    fn num_agents(&self) -> usize {
        1
    }

    fn obs_size(&self) -> usize;

    /// How many actions each head has. Every head is discrete.
    fn act_sizes(&self) -> Vec<usize>;

    /// Start fresh episodes and write the first observations
    fn reset(&mut self, agents: Agents<'_, Self::Obs>);

    /// Act on `agents.actions` and write rewards, terminals and the next observations. An
    /// agent whose episode ends is reset here, with its terminal set: the trainer does not
    /// call `reset` between episodes.
    fn step(&mut self, agents: Agents<'_, Self::Obs>);

    /// Where this env adds up its episodes, which the trainer reads and clears
    fn log(&mut self) -> &mut Log;

    /// The logs of every env as the trainer shows them: each stat averaged over the
    /// episodes since the last clear, then `n`, how many episodes that was. Few envs need
    /// anything but this.
    fn summarize(envs: &mut [Self], clear: bool) -> Vec<(String, f64)>
    where
        Self: Sized,
    {
        let mut sums: Vec<(&'static str, f64)> = Vec::new();
        let mut n = 0.0;
        for env in envs.iter_mut() {
            let log = env.log();
            if log.n != 0.0 {
                n += log.n;
                for &(key, value) in &log.sums {
                    match sums.iter_mut().find(|(k, _)| *k == key) {
                        Some(sum) => sum.1 += value,
                        None => sums.push((key, value)),
                    }
                }
            }
            if clear {
                log.clear();
            }
        }
        let mut out: Vec<(String, f64)> = Vec::new();
        if n > 0.0 {
            out.extend(sums.iter().filter(|(key, _)| *key != "n").map(|&(key, sum)| (key.to_string(), sum / n)));
        }
        out.push(("n".to_string(), n));
        out
    }

    /// Show this env, for eval. Called once a step; an env that opens a window keeps it.
    fn render(&mut self) {}

    /// Whether a window `render` opened has been closed, which ends eval
    fn window_closed(&self) -> bool {
        false
    }
}

/// An env's stats, summed over the episodes it has finished, and `n`, how many those are.
/// Stats keep the order they were first added in, which is the order the dashboard shows.
#[derive(Clone, Debug, Default)]
pub struct Log {
    pub n: f64,
    pub sums: Vec<(&'static str, f64)>,
}

impl Log {
    /// Add to one stat's sum
    pub fn add(&mut self, key: &'static str, value: f64) {
        match self.sums.iter_mut().find(|(k, _)| *k == key) {
            Some(sum) => sum.1 += value,
            None => self.sums.push((key, value)),
        }
    }

    /// One finished episode's stats
    pub fn episode(&mut self, stats: &[(&'static str, f64)]) {
        for &(key, value) in stats {
            self.add(key, value);
        }
        self.n += 1.0;
    }

    pub fn clear(&mut self) {
        self.n = 0.0;
        for sum in &mut self.sums {
            sum.1 = 0.0;
        }
    }
}

/// The [env] section of the config, as it was written there, overrides included.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pairs: Vec<(String, String)>,
}

impl Config {
    pub fn new(pairs: Vec<(String, String)>) -> Self {
        Self { pairs }
    }

    pub fn pairs(&self) -> &[(String, String)] {
        &self.pairs
    }

    /// A value as it was written, if the config has it
    pub fn text(&self, key: &str) -> Option<&str> {
        self.pairs.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// A number the config must have. 1_000 is a thousand, and true and false are 1 and 0.
    pub fn num(&self, key: &str) -> f64 {
        match self.text(key) {
            Some(raw) => parse_num(raw).unwrap_or_else(|| fail(&format!("[env] {key} should be a number, not '{raw}'"))),
            None => fail(&format!("missing key [env] {key}")),
        }
    }

    /// A number the config may leave out
    pub fn num_or(&self, key: &str, default: f64) -> f64 {
        if self.text(key).is_some() {
            self.num(key)
        } else {
            default
        }
    }

    pub fn int(&self, key: &str) -> i64 {
        self.num(key) as i64
    }

    pub fn size(&self, key: &str) -> usize {
        let value = self.num(key);
        if value < 0.0 {
            fail(&format!("[env] {key} should not be negative"));
        }
        value as usize
    }

    pub fn flag(&self, key: &str) -> bool {
        self.num_or(key, 0.0) != 0.0
    }
}

fn parse_num(raw: &str) -> Option<f64> {
    let raw = raw.trim();
    match raw.to_ascii_lowercase().as_str() {
        "true" => return Some(1.0),
        "false" => return Some(0.0),
        _ => {}
    }
    raw.replace('_', "").parse().ok()
}

/// Say what is wrong and stop, as the trainer does
pub fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_read_as_the_trainer_writes_them() {
        let config = Config::new(vec![
            ("size".into(), "11".into()),
            ("steps".into(), "1_000_000".into()),
            ("rate".into(), "2.5e-3".into()),
            ("on".into(), "True".into()),
            ("size".into(), "13".into()),
        ]);
        assert_eq!(config.size("size"), 13);
        assert_eq!(config.num("steps"), 1e6);
        assert_eq!(config.num("rate"), 2.5e-3);
        assert!(config.flag("on"));
        assert!(!config.flag("off"));
        assert_eq!(config.num_or("missing", 4.0), 4.0);
    }

    #[test]
    fn logs_average_over_episodes() {
        struct Counter(Log);
        impl Env for Counter {
            type Obs = u8;
            fn obs_size(&self) -> usize {
                1
            }
            fn act_sizes(&self) -> Vec<usize> {
                vec![2]
            }
            fn reset(&mut self, _: Agents<'_, u8>) {}
            fn step(&mut self, _: Agents<'_, u8>) {}
            fn log(&mut self) -> &mut Log {
                &mut self.0
            }
        }
        let mut a = Counter(Log::default());
        a.0.episode(&[("perf", 1.0), ("length", 10.0)]);
        a.0.episode(&[("perf", 0.0), ("length", 20.0)]);
        let mut b = Counter(Log::default());
        b.0.episode(&[("perf", 1.0), ("length", 30.0)]);
        let idle = Counter(Log::default());
        let mut envs = [a, b, idle];
        let out = Counter::summarize(&mut envs, true);
        assert_eq!(out, vec![("perf".into(), 2.0 / 3.0), ("length".into(), 20.0), ("n".into(), 3.0)]);
        assert_eq!(Counter::summarize(&mut envs, false), vec![("n".into(), 0.0)]);
    }
}
