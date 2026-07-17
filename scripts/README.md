# Scripts

## MorphoMNIST causal visualizer

Launch the Gradio app from the repository root:

```bash
python scripts/morphomnist_visualizer.py \
  --checkpoint gs://medical-airnd/causal-gen/checkpoints/t_i_d/cf_torch-gpu-g4_17-07-2026
```

Useful flags:

- `--accelerator cpu|cuda|mps|auto`
- `--server-name 127.0.0.1`
- `--server-port 7860`
- `--share` to expose a public Gradio link
