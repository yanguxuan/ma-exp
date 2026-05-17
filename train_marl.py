"""Train the Multi-Agent RL elevator dispatcher.

Run:
    python train_marl.py

The implementation uses parameter-sharing independent DQN:
- each elevator is one agent;
- all agents share the same Q-network;
- the replay buffer stores one transition per elevator per simulator step;
- the reward is a shaped team reward from system-level service quality.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.multi_agent_rl import AgentQNet, MultiAgentRLAlgorithm
from src.simulator import ElevatorSimulator


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass
class TrainConfig:
    num_floors: int = 9
    num_elevators: int = 3
    episodes: int = 50
    duration: float = 3600.0
    seed: int = 42
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    replay_size: int = 50000
    target_update_steps: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    model_path: str = "marl_model.pth"


def epsilon_by_step(cfg: TrainConfig, step: int) -> float:
    frac = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def shaped_team_reward(prev_info: dict, info: dict, state: dict) -> float:
    served_delta = info["served"] - prev_info.get("served", 0)
    boarded_delta = info["boarded"] - prev_info.get("boarded", 0)
    abandoned_delta = info["abandoned"] - prev_info.get("abandoned", 0)
    waiting_count = sum(state.get("waiting_count_by_floor", []))
    avg_wait = float(state.get("avg_wait_time", 0.0))

    return (
        20.0 * served_delta
        + 2.0 * boarded_delta
        - 50.0 * abandoned_delta
        - 0.05 * waiting_count
        - 0.01 * avg_wait
    )


def select_training_actions(
    policy_net: AgentQNet,
    state: dict,
    cfg: TrainConfig,
    device: torch.device,
    epsilon: float,
) -> list[int | None]:
    actions: list[int | None] = []
    waiting_by_floor = state.get("waiting_count_by_floor", [0] * cfg.num_floors)
    floors_with_waiting = [f for f, cnt in enumerate(waiting_by_floor) if cnt > 0]

    for agent_id, elevator in enumerate(state["elevators"]):
        load_dests = elevator.get("load_dests", [])
        if load_dests:
            current_floor = int(elevator["floor"])
            actions.append(int(min(load_dests, key=lambda f: abs(int(f) - current_floor))))
            continue

        if elevator.get("state") == "moving" and elevator.get("target_floor") is not None:
            actions.append(int(elevator["target_floor"]))
            continue

        if not floors_with_waiting and elevator.get("target_floor") is None:
            actions.append(None)
            continue

        if random.random() < epsilon:
            action = random.randrange(cfg.num_floors)
        else:
            obs = MultiAgentRLAlgorithm.extract_agent_observation(state, agent_id, cfg.num_floors)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(policy_net(obs_tensor), dim=1).item())

        if floors_with_waiting and waiting_by_floor[action] == 0:
            current_floor = int(elevator["floor"])
            action = min(floors_with_waiting, key=lambda f: abs(f - current_floor))

        actions.append(action)

    return actions


def optimize(
    policy_net: AgentQNet,
    target_net: AgentQNet,
    optimizer: optim.Optimizer,
    replay: Deque[Transition],
    cfg: TrainConfig,
    device: torch.device,
) -> float | None:
    if len(replay) < cfg.batch_size:
        return None

    batch = random.sample(replay, cfg.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_states_t).max(dim=1).values
        targets = rewards_t + cfg.gamma * next_q_values * (1.0 - dones_t)

    loss = nn.SmoothL1Loss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = MultiAgentRLAlgorithm.observation_dim(cfg.num_floors)
    policy_net = AgentQNet(input_dim, cfg.num_floors).to(device)
    target_net = AgentQNet(input_dim, cfg.num_floors).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    replay: Deque[Transition] = deque(maxlen=cfg.replay_size)
    global_step = 0

    for ep in range(cfg.episodes):
        env = ElevatorSimulator(
            num_floors=cfg.num_floors,
            num_elevators=cfg.num_elevators,
            seed=cfg.seed + ep,
        )
        state = env.reset()
        prev_info = {"served": 0, "boarded": 0, "abandoned": 0}
        ep_reward = 0.0
        losses = []

        while env.current_time < cfg.duration:
            epsilon = epsilon_by_step(cfg, global_step)
            observations = [
                MultiAgentRLAlgorithm.extract_agent_observation(state, i, cfg.num_floors)
                for i in range(cfg.num_elevators)
            ]
            actions = select_training_actions(policy_net, state, cfg, device, epsilon)
            next_state, _raw_reward, done, info = env.step(actions)
            reward = shaped_team_reward(prev_info, info, next_state)
            next_observations = [
                MultiAgentRLAlgorithm.extract_agent_observation(next_state, i, cfg.num_floors)
                for i in range(cfg.num_elevators)
            ]

            for i, action in enumerate(actions):
                if action is not None:
                    replay.append((observations[i], int(action), reward, next_observations[i], done))

            loss = optimize(policy_net, target_net, optimizer, replay, cfg, device)
            if loss is not None:
                losses.append(loss)

            global_step += 1
            if global_step % cfg.target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            ep_reward += reward
            prev_info = info
            state = next_state
            if done:
                break

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Episode {ep + 1:03d}/{cfg.episodes} "
            f"reward={ep_reward:.1f} served={prev_info['served']} "
            f"abandoned={prev_info['abandoned']} epsilon={epsilon_by_step(cfg, global_step):.3f} "
            f"loss={avg_loss:.4f}"
        )

    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "num_floors": cfg.num_floors,
            "num_elevators": cfg.num_elevators,
            "input_dim": input_dim,
            "algorithm": "parameter_sharing_independent_dqn",
        },
        cfg.model_path,
    )
    print(f"Saved MARL model to {cfg.model_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--num-floors", type=int, default=9)
    parser.add_argument("--num-elevators", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="marl_model.pth")
    args = parser.parse_args()

    return TrainConfig(
        episodes=args.episodes,
        duration=args.duration,
        num_floors=args.num_floors,
        num_elevators=args.num_elevators,
        seed=args.seed,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    train(parse_args())
