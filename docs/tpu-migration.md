# TPU Migration Plan for `causal-gen`

This document turns the current PyTorch codebase into a TPU-friendly training stack for GCP TPU v6e-4 while staying in PyTorch.

Short answer:

- Yes, you should make a few code changes before expecting a smooth TPU run.
- The models themselves are mostly PyTorch-native and do not need a rewrite.
- The main work is in runtime selection, distributed launching, dataloading, mixed precision, checkpointing, and reducing host-side bottlenecks.
- The best TPU path is PyTorch + `torch_xla` with one process per TPU core.

The goal is not to replace the project with a different framework. The goal is to keep the model code in PyTorch and make the training loops compatible with XLA execution.

## 1. Current Readiness Snapshot

The repository already has several good signs:

- the core models are written in plain PyTorch
- checkpointing is already separated from the device logic
- datasets are mostly host-side and simple
- the code already supports remote checkpoints and GCS-backed data

The main TPU blockers are:

- device selection only knows about `cpu`, `cuda`, and `mps`
- the training code assumes a single-process single-device workflow
- dataloading is tuned for CUDA, not TPU
- timing and synchronization helpers are CUDA/MPS specific
- optimizer and checkpoint restore logic assumes a single local device
- some evaluation and logging paths do extra host work every step

Relevant files:

- `src/utils.py`
- `src/train_setup.py`
- `src/main.py`
- `src/trainer.py`
- `src/benchmark.py`
- `src/pgm/train_pgm.py`
- `src/pgm/train_cf.py`
- `src/hps.py`

## 2. Migration Strategy

Use a staged approach:

1. Make the code runnable on a single TPU core first.
2. Add XLA-aware distributed execution across the 4 cores of v6e-4.
3. Convert the training loops to scale correctly with global batch size.
4. Tune the input pipeline and precision mode for TPU throughput.
5. Profile and remove host-side bottlenecks.
6. Only then start tuning model width, batch size, and accumulation for the best throughput.

This order matters because TPU performance problems are often caused by launch/runtime issues, not the model itself.

## 3. Phase 0: Environment and Runtime Setup

Before code changes, define the TPU execution model clearly.

### What to install

- a TPU-capable PyTorch build
- `torch_xla` for the matching PyTorch version
- `gcsfs` for remote dataset and checkpoint access, which the repo already expects

### What to run

Use the PyTorch/XLA launcher provided by the installed version of `torch_xla`.

The important rule is:

- one training process per TPU core
- do not run a single Python process and hope it uses all TPU cores automatically

### What to verify first

- Python can see the TPU device
- a trivial tensor move to the XLA device works
- a tiny forward/backward step works on one core
- checkpoint save/load works from the target filesystem

### Recommendation for `v6e-4`

Treat `v6e-4` as a small multi-core TPU target and design the code around data-parallel execution from day one.

## 4. Phase 1: Add a TPU Device Abstraction

The repo currently uses `select_device()` in `src/utils.py`.

### Change needed

Extend device selection so the code can choose a TPU/XLA device.

### Recommended behavior

- keep `auto`, `cpu`, and `cuda`
- add `tpu` or `xla` as an accelerator option
- map that option to the XLA device name used by your installed `torch_xla`

### Why this matters

Right now all the main scripts assume the device is a regular `torch.device`.
That is fine on CPU/GPU, but TPU execution needs a slightly different runtime contract.

### Files to update

- `src/utils.py`
- `src/hps.py`
- `src/main.py`
- `src/benchmark.py`
- `src/pgm/train_pgm.py`
- `src/pgm/train_cf.py`

### Behavior to preserve

- loading checkpoints on CPU and then moving them to the TPU device
- keeping the existing `--accelerator` flow
- making TPU an explicit option rather than silently changing semantics

## 5. Phase 2: Make the Training Loops XLA-Friendly

The training loops are mostly standard PyTorch, which is good.
The main change is to remove assumptions that only make sense on a single GPU process.

### Update the main image trainer

File: `src/trainer.py`

Key changes:

- keep the forward/backward logic in PyTorch
- ensure the loss is reduced in a way that is compatible with multi-core data parallelism
- avoid extra synchronization in the hot path
- move logging, image writing, and checkpoint sync outside the critical step path where possible

### Update the entrypoint

File: `src/main.py`

Key changes:

- initialize the TPU runtime before model creation
- wrap model and optimizer creation in the TPU-compatible launch context
- restore checkpoints to CPU first, then move weights to TPU
- move optimizer state tensors to TPU only after restore

### Update the PGM and counterfactual trainers

Files:

- `src/pgm/train_pgm.py`
- `src/pgm/train_cf.py`

Key changes:

- ensure `TraceStorage_ELBO` runs on the TPU device without hidden CPU fallbacks
- keep the Pyro parameter store consistent across replicas
- avoid Python-side control flow inside the per-step hot path where possible
- make counterfactual sampling and intervention sampling batch-friendly

### Important TPU rule

Anything that depends on the current step, batch size, or device state should be computed once per step and kept as tensor operations as much as possible.

## 6. Phase 3: Convert to True Multi-Core Data Parallelism

This is where TPU performance really starts to show up.

### Recommended execution model

Use one process per TPU core and shard the input batches across processes.

For `v6e-4`, the target is:

- 4 TPU worker processes
- one shard per core
- global batch size = per-core batch size x 4

### What this means for the code

You will need to:

- initialize the XLA process rank and world size
- shard the dataloader across replicas
- aggregate metrics across replicas
- make checkpointing happen on a single coordinator process

### Where to implement it

- a small TPU runtime helper module, or
- a lightweight distributed wrapper around the existing training entrypoints

### Things to avoid

- do not call `torch.distributed` GPU patterns directly without checking XLA support
- do not keep one global dataloader and reuse it from multiple processes without sharding
- do not let every process write its own checkpoint unless that is intentional

### Metric aggregation

Loss and metrics should be reduced across workers before you log them.
Otherwise each worker will report only its local shard and the numbers will be misleading.

## 7. Phase 4: Fix the Input Pipeline for TPU

The current dataloader settings are okay for a GPU machine, but TPU throughput depends heavily on how cleanly data arrives from the host.

### Current hotspots

File: `src/train_setup.py`

Current behavior:

- `num_workers` is tied to local CPU count
- `pin_memory` is enabled only for CUDA
- training and eval loaders are created as single-process loaders

### Recommended TPU changes

- use a TPU-appropriate worker count, not `os.cpu_count() // 2` by default
- tune `num_workers`, `prefetch_factor`, and `persistent_workers`
- keep CPU preprocessing light and deterministic
- avoid expensive Python logic in `__getitem__`
- minimize per-sample file opens if you can cache metadata or pre-index data

### Dataset-specific notes

File: `src/datasets.py`

- image loading from GCS and local files is fine conceptually, but may become a bottleneck
- host-side PIL decode is okay, but if throughput is low you should consider caching or local staging
- transformations should stay simple and stateless

### Practical goal

Feed the TPU steadily enough that it is not waiting on Python image loading.

## 8. Phase 5: Use bfloat16 Where It Helps

TPUs are strongest when the training path uses bfloat16-friendly math.

### Recommended precision plan

- keep model weights and optimizer state in the usual PyTorch layout
- use bfloat16 for the forward/backward compute path where the runtime supports it
- keep numerically sensitive reductions in FP32 if needed

### Why bfloat16 is a good fit

- it preserves a wide numeric range
- it usually requires less hyperparameter retuning than FP16
- it is the natural low-precision path for TPU execution

### What to check in this repo

- the discretized Gaussian likelihood in `src/vae.py` and `src/simple_vae.py`
- the logistic likelihood in `src/dmol.py`
- KL and NLL reductions in `src/trainer.py`
- Pyro distribution code in `src/pgm/layers.py` and `src/pgm/flow_pgm.py`

### Safety rule

If a term is prone to underflow or overflow, keep that specific reduction in FP32 even if the surrounding model uses bfloat16.

## 9. Phase 6: Reduce Host Synchronization

TPU performance drops fast if the host keeps forcing synchronization.

### Current sync points to audit

- `torch.cuda.synchronize()` in `src/benchmark.py`
- CPU-side `.item()` calls in the hot path
- frequent TensorBoard writes
- frequent checkpoint syncs to remote storage
- repeated `torch.randperm` on the host for counterfactual setup

### Recommended changes

- replace GPU-specific sync calls with TPU-safe timing logic
- only call `.item()` when you are done accumulating a metric for logging
- batch logging every N steps instead of every step where possible
- save checkpoints less frequently than scalar logs
- precompute or batch intervention indices when possible

### Why this matters

TPU kernels are fast, but the host can become the bottleneck if the loop is chatty.

## 10. Phase 7: Make Checkpointing TPU-Safe

Checkpointing already uses a CPU-first load pattern, which is a good start.
You still need a TPU-specific write strategy.

### Current code path

- checkpoints are loaded on CPU with `map_location="cpu"`
- optimizer tensors are manually moved after restore
- model state is saved from the live model
- remote sync is handled via local staging and GCS copy helpers

### Recommended TPU checkpoint policy

- only rank 0 writes checkpoints
- save CPU copies of model and optimizer state when practical
- restore on CPU first, then move to TPU
- keep the checkpoint format backward compatible with current files

### Files to revisit

- `src/main.py`
- `src/pgm/train_pgm.py`
- `src/pgm/train_cf.py`
- `src/utils.py`

### Extra recommendation

Store the TPU world size, process rank, and accelerator type in the checkpoint metadata so resuming is unambiguous.

## 11. Phase 8: Parallelism Tuning While Staying in PyTorch

Once the code runs correctly, optimize for throughput.

### Highest-value knobs

1. Increase global batch size until you saturate the TPU.
2. Use gradient accumulation only if memory forces you to.
3. Keep per-step Python overhead low.
4. Profile the input pipeline separately from the model step.
5. Use multi-core data parallelism before adding more complicated model parallel ideas.

### TPU-friendly parallelism plan

- data parallelism first
- model parallelism only if the model no longer fits
- avoid splitting the hierarchical VAE across devices unless memory becomes the limiting factor

### Why data parallelism is the right first step

The current models are moderate-sized convolutional and MLP-heavy systems.
That is a good fit for simple data parallel execution on a small TPU slice.

## 12. Phase 9: Model-Specific Optimization Guidance

### For `src/vae.py` and `src/simple_vae.py`

Focus on:

- bigger effective batch size
- bfloat16-safe math
- fewer host round-trips during sampling
- minimizing Python branching in `forward` and `sample`

Potential follow-ups:

- consider `torch.compile` only after the XLA path is stable and benchmarked
- keep tensor shapes as static as possible
- avoid per-sample `if` branches in the hot path

### For `src/pgm/flow_pgm.py`

Focus on:

- reducing overhead in Pyro model and guide execution
- keeping particle-based ELBO computation efficient
- batching all repeated probability calculations

### For `src/pgm/train_cf.py`

Focus on:

- batching do-interventions
- removing unnecessary deep copies in the training loop
- using rank-local randomness carefully so replicas stay independent but reproducible

## 13. Phase 10: TPU Benchmarking and Validation

Before claiming success, validate in this order:

1. one forward/backward step on one TPU core
2. one short epoch on one core
3. four-core training on a tiny subset
4. full training with the intended batch size
5. checkpoint save and resume
6. evaluation and sampling

### What to compare

- step time
- samples per second
- host utilization
- TPU utilization
- loss stability
- reconstruction quality
- counterfactual quality

### Update `src/benchmark.py`

Make the benchmark TPU-aware so it measures the TPU path, not just CUDA/MPS behavior.

Specific changes:

- remove CUDA-specific synchronization
- add XLA-safe synchronization if needed for accurate timing
- benchmark both per-core and global throughput

## 14. Suggested Implementation Order

If we were doing this as a code migration, I would do it in this order:

1. Add TPU device selection and launcher support.
2. Make `src/main.py` run one step on one TPU core.
3. Update `src/train_setup.py` for TPU-friendly dataloaders.
4. Make checkpoint save/load work reliably on TPU.
5. Add XLA-aware timing and logging.
6. Port `src/pgm/train_pgm.py` and `src/pgm/train_cf.py`.
7. Add replica-aware metric aggregation.
8. Tune global batch size and accumulation.
9. Run end-to-end benchmarks.
10. Only then tune precision and architectural knobs.

## 15. Concrete Risk List

The main risks are:

- hidden CPU fallbacks in the training step
- too much host-side logging
- dataloader starvation
- optimizer state not restored on the right device
- over-aggressive low precision in numerically sensitive likelihood terms
- forgetting to shard data across TPU workers

## 16. Bottom Line

You do not need to rewrite the model architecture to use TPU.
You do need to make the training stack explicit about:

- TPU device selection
- XLA launch semantics
- distributed batch sharding
- bfloat16-friendly math
- checkpointing and logging coordination

That is enough to make the repo TPU-compatible while staying fully in PyTorch.

