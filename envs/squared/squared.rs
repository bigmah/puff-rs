//! Squared, PufferLib's first env (ocean/squared), in Rust: an agent in the middle of a
//! size x size grid walks to a target somewhere else on it. Reaching the target is +1;
//! walking off the grid, or taking more than 3 * size steps, is -1. Either ends the
//! episode.
//!
//! Observations: one byte a cell, 0 empty, 1 the agent, 2 the target.
//! Actions: one head, 0 stay, 1 down, 2 up, 3 left, 4 right.
//!
//! The target is drawn from this env's own generator, seeded by its index, so a run
//! repeats but does not draw the C env's targets.

use puffer_env::{Agents, Config, Env, Log};

const EMPTY: u8 = 0;
const AGENT: u8 = 1;
const TARGET: u8 = 2;

pub struct Squared {
    size: usize,
    tick: usize,
    r: isize,
    c: isize,
    rng: u64,
    log: Log,
    /// The last observation, for render
    grid: Vec<u8>,
}

impl Squared {
    pub fn new(config: &Config, index: usize) -> Self {
        let size = config.size("size");
        if size < 2 {
            puffer_env::fail("[env] size should be at least 2");
        }
        Self {
            size,
            tick: 0,
            r: 0,
            c: 0,
            // splitmix64 from the index, so neighbouring envs start far apart
            rng: (index as u64).wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            log: Log::default(),
            grid: vec![EMPTY; size * size],
        }
    }

    fn next(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn start(&mut self, obs: &mut [u8]) {
        let tiles = self.size * self.size;
        obs.fill(EMPTY);
        obs[tiles / 2] = AGENT;
        self.r = (self.size / 2) as isize;
        self.c = (self.size / 2) as isize;
        self.tick = 0;
        let target = loop {
            let target = (self.next() % tiles as u64) as usize;
            if target != tiles / 2 {
                break target;
            }
        };
        obs[target] = TARGET;
    }

    fn end(&mut self, agents: &mut Agents<'_, u8>, reward: f32) {
        agents.terminals[0] = 1.0;
        agents.rewards[0] = reward;
        self.log.episode(&[
            ("perf", if reward > 0.0 { 1.0 } else { 0.0 }),
            ("score", reward as f64),
            ("episode_return", reward as f64),
            ("episode_length", self.tick as f64),
        ]);
        self.start(agents.observation(0));
    }
}

impl Env for Squared {
    type Obs = u8;

    fn obs_size(&self) -> usize {
        self.size * self.size
    }

    fn act_sizes(&self) -> Vec<usize> {
        vec![5]
    }

    fn reset(&mut self, mut agents: Agents<'_, u8>) {
        self.start(agents.observation(0));
        self.grid.copy_from_slice(agents.observation(0));
    }

    fn step(&mut self, mut agents: Agents<'_, u8>) {
        self.tick += 1;
        let size = self.size as isize;
        let obs = agents.observation(0);
        obs[(self.r * size + self.c) as usize] = EMPTY;
        match agents.action(0, 0) {
            1 => self.r += 1,
            2 => self.r -= 1,
            3 => self.c -= 1,
            4 => self.c += 1,
            _ => {}
        }

        let off = self.r < 0 || self.c < 0 || self.r >= size || self.c >= size;
        if off || self.tick > 3 * self.size {
            self.end(&mut agents, -1.0);
        } else {
            let pos = (self.r * size + self.c) as usize;
            let obs = agents.observation(0);
            if obs[pos] == TARGET {
                self.end(&mut agents, 1.0);
            } else {
                obs[pos] = AGENT;
            }
        }
        self.grid.copy_from_slice(agents.observation(0));
    }

    fn log(&mut self) -> &mut Log {
        &mut self.log
    }

    /// The grid in the terminal, ten frames a second: @ the agent, X the target
    fn render(&mut self) {
        let mut frame = String::from("\x1b[H\x1b[2J");
        for row in self.grid.chunks(self.size) {
            for &cell in row {
                frame.push_str(match cell {
                    AGENT => " @",
                    TARGET => " X",
                    _ => " .",
                });
            }
            frame.push('\n');
        }
        print!("{frame}");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(env: &mut Squared, obs: &mut [u8], action: f32, reset: bool) -> (f32, f32) {
        let (mut r, mut t, mut m) = ([0.0], [0.0], [1u8; 5]);
        let actions = [action];
        let agents = Agents { observations: obs, actions: &actions, rewards: &mut r, terminals: &mut t, action_mask: &mut m, obs_size: 25, num_atns: 1, mask_size: 5 };
        if reset { env.reset(agents) } else { env.step(agents) }
        (r[0], t[0])
    }

    #[test]
    fn walking_to_the_target_pays_and_walking_off_costs() {
        let config = Config::new(vec![("size".into(), "5".into())]);
        let mut env = Squared::new(&config, 0);
        let mut obs = [0u8; 25];
        run(&mut env, &mut obs, 0.0, true);
        assert_eq!(obs.iter().filter(|&&c| c == AGENT).count(), 1);
        let target = obs.iter().position(|&c| c == TARGET).unwrap() as isize;
        let (tr, tc) = (target / 5, target % 5);

        // Down or up, then left or right, to the target
        let mut moves = Vec::new();
        moves.extend(std::iter::repeat_n(if tr > 2 { 1.0 } else { 2.0 }, (tr - 2).unsigned_abs()));
        moves.extend(std::iter::repeat_n(if tc > 2 { 4.0 } else { 3.0 }, (tc - 2).unsigned_abs()));
        let last = moves.pop().unwrap();
        for action in moves {
            assert_eq!(run(&mut env, &mut obs, action, false), (0.0, 0.0));
        }
        assert_eq!(run(&mut env, &mut obs, last, false), (1.0, 1.0));
        assert_eq!(env.log.n, 1.0);

        // A fresh episode began; with the target in a corner out of the way, three steps up
        // from the middle of five is off the top
        assert_eq!(obs.iter().filter(|&&c| c == AGENT).count(), 1);
        obs.fill(EMPTY);
        obs[12] = AGENT;
        obs[24] = TARGET;
        (env.r, env.c) = (2, 2);
        assert_eq!(run(&mut env, &mut obs, 2.0, false), (0.0, 0.0));
        assert_eq!(run(&mut env, &mut obs, 2.0, false), (0.0, 0.0));
        assert_eq!(run(&mut env, &mut obs, 2.0, false), (-1.0, 1.0));
        assert_eq!(env.log.n, 2.0);
    }
}
