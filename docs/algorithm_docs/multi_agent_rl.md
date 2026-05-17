# Multi-Agent RL Elevator Dispatching

## Position in this project

`Multi-Agent RL` is the third dispatching method in this project, parallel to:

- `ETA-based`: rule/ETA industrial baseline.
- `IL+RL`: imitation learning plus reinforcement learning with one global policy.
- `Multi-Agent RL`: one elevator is one agent; agents share a Q-network and make decentralized decisions.

## Files

- `src/algorithms/multi_agent_rl.py`: inference-time algorithm class.
- `train_marl.py`: training script for parameter-sharing independent DQN.
- `compare_all_methods.py`: evaluates ETA-based, IL+RL, and Multi-Agent RL together.
- `marl_model.pth`: generated after training; not required for running because the algorithm has a deterministic fallback.

## Method

The method treats the elevator group as a cooperative multi-agent system.
Each elevator observes:

- its own floor, direction, load, state, and target;
- waiting passenger counts by floor;
- global queue pressure, average waiting time, and episode time.

All elevators use the same `AgentQNet`, so training data from every elevator
updates one shared policy. At each simulator step, every elevator agent selects
a target floor. The simulator then executes the joint action.

## Reward

Training uses a shaped team reward:

- positive reward for served passengers;
- smaller positive reward for boarded passengers;
- large penalty for abandoned passengers;
- small continuous penalties for queue length and average waiting time.

This keeps the learning signal denser than the simulator's sparse service-only
reward and encourages the whole elevator group to reduce waiting pressure.

## Usage

Train:

```bash
python train_marl.py --episodes 50 --duration 3600 --model-path marl_model.pth
```

Quick smoke training:

```bash
python train_marl.py --episodes 2 --duration 300 --model-path marl_model.pth
```

Compare all three methods:

```bash
python compare_all_methods.py
```

Or use the existing runner:

```bash
python -m src.experiment_runner
```

## Notes

If `marl_model.pth` does not exist, `MultiAgentRLAlgorithm` still runs by using
a deterministic heuristic fallback. This makes the new method pluggable before
training, while preserving the same `BaseAlgorithm.select_actions(state)` API.
