# Multi-Agent RL Extension

This project now includes a third elevator dispatching method, parallel to
`ETA-based` and `IL+RL`: `Multi-Agent RL`.

## Added Files

- `src/algorithms/multi_agent_rl.py`
- `train_marl.py`
- `compare_all_methods.py`
- `docs/algorithm_docs/multi_agent_rl.md`

## Train

```bash
python train_marl.py --episodes 50 --duration 3600 --model-path marl_model.pth
```

Quick smoke training:

```bash
python train_marl.py --episodes 2 --duration 300 --model-path marl_model.pth
```

## Compare

```bash
python compare_all_methods.py
```

The old comparison entry also works:

```bash
python compare_eta_vs_ilrl.py
```

## Runner

```bash
python -m src.experiment_runner
```

If `marl_model.pth` is missing, `MultiAgentRLAlgorithm` uses a deterministic
fallback policy so the project remains runnable before training.
