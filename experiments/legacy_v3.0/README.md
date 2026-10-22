# Legacy v3.0 artefacts — DO NOT LOAD OR CITE

Weights and resume state produced before the v3.1 corrections (different generator parametrisation, encoder,
split and protocol). They are archived only for provenance. The current code rejects them: checkpoints, run
state and live contracts carry `PROTOCOL_VERSION` (see `experiments/qspinn_core.py`) and are ignored on
mismatch. Never pad or reshape these tensors to make them load.
