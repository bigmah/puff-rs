//! The vecenv: PufferLib 5.0's CPU vecenv (env_setup and the worker loop in
//! src/pufferl.cu) for any [`Env`]. Envs are made in order until there are total_agents
//! agents, each told its index, and step together on num_threads threads. The
//! observations, actions, rewards, terminals and action masks of every agent are flat
//! arrays here, which each env sees its own agents' slices of.
//!
//! An env is anything that implements [`Env`]: a Rust env, or one behind a bridge to
//! another language that makes an [`Env`] of it. The stepping, threads and logs here are
//! the same for every kind.

use puffer_env::{Agents, Env, Obs};
use rayon::prelude::*;

/// A vecenv of envs of one kind, whatever kind that is
trait Vectorized {
    fn reset(&mut self);
    fn step(&mut self);
    fn observations(&self, out: &mut Vec<f32>);
    fn rewards(&self) -> &[f32];
    fn terminals(&self) -> &[f32];
    fn set_actions(&mut self, actions: &[u32]);
    fn log(&mut self, clear: bool) -> Vec<(String, f64)>;
    fn render(&mut self, env_id: usize);
    fn window_closed(&self) -> bool;
    fn close(&mut self);
}

pub struct VecEnv {
    pub total_agents: usize,
    pub obs_size: usize,
    pub act_sizes: Vec<usize>,
    inner: Box<dyn Vectorized>,
}

impl VecEnv {
    /// Envs made by make(0), make(1), ... until they have total_agents agents between them
    pub fn from_envs<E: Env + 'static>(make: impl FnMut(usize) -> E, total_agents: usize, num_threads: usize) -> Self {
        let envs = Envs::new(make, total_agents, num_threads);
        Self {
            total_agents,
            obs_size: envs.obs_size,
            act_sizes: envs.act_sizes.clone(),
            inner: Box::new(envs),
        }
    }

    /// Fresh episodes everywhere, as env_restart starts a run
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// One step of every env, on the actions in place.
    pub fn step(&mut self) {
        self.inner.step();
    }

    /// Every agent's observation as floats, which is what the policy's first matrix takes.
    pub fn observations(&self, out: &mut Vec<f32>) {
        self.inner.observations(out);
    }

    pub fn rewards(&self) -> &[f32] {
        self.inner.rewards()
    }

    pub fn terminals(&self) -> &[f32] {
        self.inner.terminals()
    }

    /// The env reads its actions as floats, one for each head of each agent.
    pub fn set_actions(&mut self, actions: &[u32]) {
        self.inner.set_actions(actions);
    }

    /// Env stats averaged over the episodes since the last clear, and n, how many
    pub fn log(&mut self, clear: bool) -> Vec<(String, f64)> {
        self.inner.log(clear)
    }

    pub fn render(&mut self, env_id: usize) {
        self.inner.render(env_id);
    }

    pub fn window_closed(&self) -> bool {
        self.inner.window_closed()
    }

    pub fn close(&mut self) {
        self.inner.close();
    }
}

impl Drop for VecEnv {
    fn drop(&mut self) {
        self.close();
    }
}

struct Envs<E: Env> {
    envs: Vec<E>,
    /// How many agents each env has, in order
    agents: Vec<usize>,
    obs_size: usize,
    act_sizes: Vec<usize>,
    mask_size: usize,
    observations: Vec<E::Obs>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    terminals: Vec<f32>,
    action_mask: Vec<u8>,
    /// None for one thread, which steps envs on the thread that asks
    pool: Option<rayon::ThreadPool>,
    threads: usize,
}

impl<E: Env + 'static> Envs<E> {
    fn new(mut make: impl FnMut(usize) -> E, total_agents: usize, num_threads: usize) -> Self {
        let mut envs = Vec::new();
        let mut agents = Vec::new();
        let mut count = 0;
        while count < total_agents {
            let env = make(envs.len());
            let n = env.num_agents();
            if n == 0 {
                crate::fail("vecenv: an env with no agents");
            }
            count += n;
            agents.push(n);
            envs.push(env);
        }
        if count != total_agents {
            crate::fail(&format!("vecenv: {} envs make {count} agents, not the {total_agents} asked for", envs.len()));
        }
        // Only final once the envs are made: an env may size its observation from its config
        let obs_size = envs[0].obs_size();
        let act_sizes = envs[0].act_sizes();
        if let Some(i) = envs.iter().position(|env| env.obs_size() != obs_size || env.act_sizes() != act_sizes) {
            crate::fail(&format!("vecenv: env {i}'s observations or actions are not the size env 0's are"));
        }
        let mask_size = act_sizes.iter().sum();
        let threads = num_threads.max(1);
        let pool = (threads > 1).then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("vecenv-{i}"))
                .build()
                .unwrap_or_else(|e| crate::fail(&format!("vecenv: {e}")))
        });
        Self {
            envs,
            agents,
            obs_size,
            mask_size,
            observations: vec![E::Obs::default(); total_agents * obs_size],
            actions: vec![0.0; total_agents * act_sizes.len()],
            rewards: vec![0.0; total_agents],
            terminals: vec![0.0; total_agents],
            action_mask: vec![1; total_agents * mask_size],
            act_sizes,
            pool,
            threads,
        }
    }

    /// f on every env with its agents' slices, on the vecenv's threads
    fn each(&mut self, f: fn(&mut E, Agents<'_, E::Obs>)) {
        self.rewards.fill(0.0);
        self.terminals.fill(0.0);
        let (obs_size, num_atns, mask_size) = (self.obs_size, self.act_sizes.len(), self.mask_size);
        let mut observations = self.observations.as_mut_slice();
        let mut actions = self.actions.as_slice();
        let mut rewards = self.rewards.as_mut_slice();
        let mut terminals = self.terminals.as_mut_slice();
        let mut action_mask = self.action_mask.as_mut_slice();
        let mut work = Vec::with_capacity(self.envs.len());
        for (env, &n) in self.envs.iter_mut().zip(&self.agents) {
            let (o, rest) = std::mem::take(&mut observations).split_at_mut(n * obs_size);
            observations = rest;
            let (a, rest) = actions.split_at(n * num_atns);
            actions = rest;
            let (r, rest) = std::mem::take(&mut rewards).split_at_mut(n);
            rewards = rest;
            let (t, rest) = std::mem::take(&mut terminals).split_at_mut(n);
            terminals = rest;
            let (m, rest) = std::mem::take(&mut action_mask).split_at_mut(n * mask_size);
            action_mask = rest;
            let agents = Agents {
                observations: o,
                actions: a,
                rewards: r,
                terminals: t,
                action_mask: m,
                obs_size,
                num_atns,
                mask_size,
            };
            work.push((env, agents));
        }
        match &self.pool {
            None => work.into_iter().for_each(|(env, agents)| f(env, agents)),
            Some(pool) => {
                // A few runs of envs a thread, so a slow env holds up less than its share
                let min_len = (work.len() / (self.threads * 4)).max(1);
                pool.install(|| work.into_par_iter().with_min_len(min_len).for_each(|(env, agents)| f(env, agents)));
            }
        }
    }
}

impl<E: Env + 'static> Vectorized for Envs<E> {
    fn reset(&mut self) {
        self.each(|env, agents| env.reset(agents));
    }

    fn step(&mut self) {
        self.each(|env, agents| env.step(agents));
    }

    fn observations(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.observations.iter().map(|&o| o.to_f32()));
    }

    fn rewards(&self) -> &[f32] {
        &self.rewards
    }

    fn terminals(&self) -> &[f32] {
        &self.terminals
    }

    fn set_actions(&mut self, actions: &[u32]) {
        for (slot, &action) in self.actions.iter_mut().zip(actions) {
            *slot = action as f32;
        }
    }

    fn log(&mut self, clear: bool) -> Vec<(String, f64)> {
        E::summarize(&mut self.envs, clear)
    }

    fn render(&mut self, env_id: usize) {
        if let Some(env) = self.envs.get_mut(env_id) {
            env.render();
        }
    }

    fn window_closed(&self) -> bool {
        self.envs.first().is_some_and(|env| env.window_closed())
    }

    /// Envs close as they drop
    fn close(&mut self) {
        self.envs.clear();
        self.agents.clear();
    }
}
