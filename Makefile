.PHONY: build test bench train eval

# Extra flags pass through, e.g. make train ARGS="--train.total-timesteps 10_000_000"
SQUARED = cargo run --release --example squared --

build:
	cargo build --release --workspace --examples

# The kernels against the CPU, acting against training, the whole of a training step
# against golden numbers (tests/golden), and the env API
test:
	cargo test --release --workspace

# Where the time goes on this GPU, with no env
bench:
	cargo run --release --example bench_matmul
	cargo run --release --example bench_train
	cargo run --release --example bench_latency

# The example env, Squared: train it, then watch the most trained checkpoint play
train:
	$(SQUARED) train squared $(ARGS)

eval:
	$(SQUARED) eval squared latest $(ARGS)
