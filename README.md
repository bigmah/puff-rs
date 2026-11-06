# puff-rs

[PufferLib](https://github.com/PufferAI/PufferLib) 5.0 in Rust: its trainer on [wgpu](https://wgpu.rs), so it runs on whatever GPU the machine has (Metal, Vulkan or DX12), and its vecenv, for envs written in Rust. No CUDA, no Python, no C, no autograd. Every forward and backward pass is a WGSL compute kernel.

The envs it has been used for, Super Mario 64 and the rest, are in [puff-rs-envs](https://github.com/bigmah/puff-rs-envs).

## Quick start

You need Rust ([rustup](https://rustup.rs)) and a GPU wgpu can use.

```sh
cargo run --release --example squared -- train squared   # about ten seconds
cargo run --release --example squared -- eval squared latest
```

`examples/squared/` is PufferLib's first env, Squared, as a whole program: an agent walks to a target on a grid. `make train` and `make eval` do the same.

## Your own env

An env implements `pufferl::env::Env`:

```rust
use pufferl::env::{Agents, Config, Env, Log};

pub struct MyEnv { log: Log, /* ... */ }

impl MyEnv {
    // config is the [env] section of the config; index is which of the envs this is
    pub fn new(config: &Config, index: usize) -> Self { /* ... */ }
}

impl Env for MyEnv {
    type Obs = u8;                                     // or f32
    fn obs_size(&self) -> usize { 121 }
    fn act_sizes(&self) -> Vec<usize> { vec![5] }      // one discrete head of 5
    fn reset(&mut self, agents: Agents<u8>) { /* write the first observation */ }
    fn step(&mut self, agents: Agents<u8>) {
        // read agents.action(0, 0), write agents.rewards, agents.terminals and the next
        // observation; at an episode's end, self.log.episode(&[("score", ...)]) and reset
        // that agent here
    }
    fn log(&mut self) -> &mut Log { &mut self.log }
}
```

A program adds its envs to an `Envs` by name, each with its default config, and hands that to the command line:

```rust
fn main() {
    pufferl::cli::main(pufferl::Envs::new()
        .add("my_env", include_str!("my_env.ini"), MyEnv::new));
}
```

```toml
[dependencies]
puff-rs = { git = "https://github.com/bigmah/puff-rs" }
```

That's all: `cargo run --release -- train my_env`. The vecenv makes envs until it has `vec.total_agents` agents, owns the observation, action, reward and terminal buffers, hands each env its own agents' slices of them, steps the envs together on `vec.num_threads` threads, and averages their logs. An env with several agents says so in `num_agents`. `render` is for eval, and is the env's own to draw however it likes.

An env can be anything behind the trait, a C env through a bridge included: [puff-rs-envs](https://github.com/bigmah/puff-rs-envs) runs PufferLib's C envs that way, with an `Envs::fallback` that finds them on disk.

The trait alone is the `puffer-env` crate (`crates/puffer-env`, no dependencies), for an env crate that shouldn't depend on the trainer.

## Layout

```
src/                  the trainer and the vecenv; src/kernels/ holds the WGSL compute kernels
crates/puffer-env/    the Env trait, Config and Log (pufferl::env)
config/default.ini    PufferLib 5.0's default config, built in
examples/squared/     a whole program: an env, its ini, and main
examples/bench_*.rs   GPU benchmarks (make bench)
tests/                kernels against the CPU, acting against training, golden numbers
```

## The trainer

PufferLib 5.0 trains with one CUDA program (`pufferl.cu` and `algo.cu`), which only an NVIDIA card runs. `src/` is that trainer on wgpu, with each kernel written once in WGSL.

**What it trains** is 5.0's:

- **Policy.** PufferLib's default: a bias-free Linear encoder, a 4-layer MinGRU with hidden size 128, and one decoder matrix whose last row is the value head. If the env's config names a `picture_width` and `picture_height`, the tail of the observation is an RGB picture, and the Nature DQN's three convolutions go in front of the encoder.
- **PPO as 5.0 has it.** Each minibatch computes advantages from its own values before the update (GAE, or V-trace with `vtrace = 1`) and doesn't normalize them. Minibatches are whole agent rows, with no prioritized replay. The MinGRU's state carries across horizons and clears where an episode ends.
- **Muon**, on the clipped gradient.

**Config** works as in 5.0. Values are read in this order, later over earlier:

1. PufferLib's `default.ini` (`config/default.ini`, built in)
2. the env's own ini, as it was added to `Envs`
3. each `--config FILE`, in order
4. `--section.key=value` flags (a space works as well as `=`, and dashes in a key are underscores)

A `--config` file or a flag can only set a key an earlier config already has, so a misspelt key is an error rather than a setting nobody reads. `--config` is this repo's addition: a variant of an env, like one level of it.

**Output.** Checkpoints go to `checkpoints/<env>/<run_id>/<agent step>.bin`. When a run ends it writes `logs/<env>/<run_id>.ini`, which holds the config and a downsampled `[metrics]` section. `latest` is the checkpoint with the most agent steps, not the newest file. `eval` shows one env in a window, or with `--headless` plays `eval_episodes` episodes and prints the score.

Checkpoints are in 5.0's format, flat float32 weights in the CUDA trainer's order and padding, so PufferLib's own CPU eval (`puffercpu.c`) reads what this writes. Checkpoints from this repo's earlier 4.0 port (numpy `.npz` data under the same `.bin` names) still load.

**Additions to 5.0**, all off unless a config asks. The keys aren't in PufferLib's `default.ini`, so an env's ini has to define one before a flag can set it:

| key in `[train]` | what it does |
|---|---|
| `norm_adv` | normalize each minibatch's advantages, as PufferLib 4.0 did |
| `weight_decay` | keeps Muon's weights bounded while there's no reward to learn from |
| `target_entropy`, `ent_coef_rate` | hold entropy at a target by moving `ent_coef`, instead of paying a fixed price |
| `ent_coef_damping` | adds the entropy gap itself to the coefficient, which stops the controller cycling once a run scores often |

**Not ported:** continuous actions, action masks, multiple GPUs, self-play and sweeps. Rollouts are always synchronous, so 5.0's `async = 1` is ignored.

### How it's built

wgpu runs compute shaders and nothing else: no BLAS, no tensors, no gradients. `algo.cu` has no autograd either. It is about 35 CUDA kernels with their backward passes written out, plus cuBLAS. This has the same shape, with one matmul kernel in place of cuBLAS:

- `matmul.wgsl`: each thread computes a 4×4 block of the result, and a long inner dimension is cut into chunks that run side by side and are summed after. Matrices are views (an offset and two strides), so a transpose or a block of the flat weight buffer costs nothing.
- `scan_forward.wgsl`, `scan_backward.wgsl` and `mingru_step.wgsl`: the MinGRU over a segment, its backward pass, and the single step that acting takes.
- `ppo_pre.wgsl`, `advantage.wgsl` and `ppo_loss.wgsl`: the loss of each step and, in the same walk, its gradient with respect to the logits and value.
- `muon_momentum.wgsl`, `ns_init.wgsl`, `ns_poly.wgsl` and `muon_apply.wgsl`: Muon, with five Newton-Schulz steps. The MinGRU's layers are one shape, so they're orthogonalized as one batch.
- `im2col.wgsl` and `col2im.wgsl`: the picture's convolutions as matmuls.

Every shape is known when training starts, so every dispatch is built once: its pipeline, its buffers, and its parameters in a uniform buffer of its own. A minibatch is a list of 146 dispatches that is encoded again each time, which takes the CPU 0.2 ms. All the weights live in one buffer in a checkpoint's layout, so saving a checkpoint is writing that buffer out. Nothing is read back while training except the actions, once a step, and eight loss statistics, once an epoch.

### Tests and benchmarks

`make test` runs:

- `tests/parity.rs`: a policy and a minibatch against golden numbers in `tests/golden/`: the loss statistics, every gradient, and the weights after one and two Muon steps. Three cases cover 5.0 as it is, everything a config can turn on, and a picture through the convolutions. The numbers came from an MLX port of 5.0's trainer, whose gradients were autograd's. That port is gone from the repo, so the golden files are frozen: a change to *what* the trainer computes has nothing left to check its backward pass against.
- `tests/rollout.rs`: acting and training are different kernels over the same policy, so a rollout trained on before the weights move must come out exactly on-policy (KL 0, importance 1). Sampling is checked against the distribution it drew from.
- `tests/ops.rs`: the matmul and the reductions against the CPU, over every way the trainer uses them.

`make bench` measures the matmul, a training epoch with no env, and the round trip to the GPU, on the GPU it's run on.

### Speed and portability

On an M4 Pro, Squared trains at 1.85M steps a second with 4,096 agents, and puff-rs-envs's platformer at 455K with 1,024, where 20M steps take 45 seconds and reach the flag in 99.7% of 2,000 episodes. There the matmul is what takes the time, at 1.5 to 2 TFLOPS with no vendor library under it. A step of acting costs 0.22 ms against the 0.18 ms a GPU round trip costs with nothing to do.

The trainer has been run on Metal, and on Vulkan on arm64 Linux, where `make test` passes against the same golden numbers. x86-64 Linux is written for and hasn't been run. `WGPU_BACKEND` and `WGPU_ADAPTER_NAME` pick a GPU, as for any wgpu program. With no GPU at all it still trains on Mesa's CPU Vulkan driver (`libvulkan1 mesa-vulkan-drivers`), slower rather than stuck.

