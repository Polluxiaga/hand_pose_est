Vendored minimal TAPIR subset.

This directory keeps only the PyTorch TAPIR inference files used by
`tapir_match_core.py`:

- `tapnet/torch/tapir_model.py`
- `tapnet/torch/nets.py`
- `tapnet/torch/utils.py`

Original source: Google DeepMind TAPNet, Apache-2.0. See `LICENSE`.

Expected checkpoint location:

- `tapnet/checkpoints/tapir_checkpoint_panning.pt`

The pipeline also accepts an explicit path via `--tapir-checkpoint`.
