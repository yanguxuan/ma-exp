"""Multi-agent reinforcement learning elevator dispatcher.

This module adds a third algorithm family beside ETA-based and IL+RL.
Each elevator is treated as an independent agent. Agents share one Q-network
so the policy can generalize across elevators, while every agent receives an
observation containing its own state plus compact global traffic information.

The class can be used in two modes:
- with a trained checkpoint: greedy decentralized Q-value control;
- without a checkpoint: a deterministic MARL-compatible heuristic fallback.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAlgorithm


class AgentQNet(nn.Module):
    """Shared Q-network used by all elevator agents."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiAgentRLAlgorithm(BaseAlgorithm):
    """Parameter-sharing independent DQN for elevator group control."""

    def __init__(
        self,
        num_elevators: int,
        num_floors: int,
        model_path: Optional[str] = "marl_model.pth",
        hidden_dim: int = 128,
        epsilon: float = 0.0,
        heuristic_fallback: bool = True,
    ):
        super().__init__("Multi-Agent RL", num_elevators, num_floors)
        self.num_actions = num_floors
        self.input_dim = self.observation_dim(num_floors)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.heuristic_fallback = heuristic_fallback
        self.rng = np.random.RandomState(0)

        self.policy_net = AgentQNet(self.input_dim, self.num_actions, hidden_dim).to(self.device)
        self.model_loaded = False

        if model_path:
            self.load_model(model_path)

    @staticmethod
    def observation_dim(num_floors: int) -> int:
        # 7 local elevator features + num_floors waiting-count features
        # + 3 global traffic features.
        return 7 + num_floors + 3

    def load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            return

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()
        self.model_loaded = True
        print(f"[MARL] Loaded model from {model_path}")

    def reset(self):
        # Keep deterministic inference behavior episode to episode.
        self.rng = np.random.RandomState(0)

    def select_actions(self, state: Dict) -> List[Optional[int]]:
        if not self.model_loaded and self.heuristic_fallback:
            return self._heuristic_actions(state)

        actions: List[Optional[int]] = []
        waiting_by_floor = state.get("waiting_count_by_floor", [0] * self.num_floors)
        floors_with_waiting = [f for f, cnt in enumerate(waiting_by_floor) if cnt > 0]

        for agent_id in range(self.num_elevators):
            elevator = state["elevators"][agent_id]

            # Passengers already inside the elevator are served first.
            load_dests = elevator.get("load_dests", [])
            if load_dests:
                current_floor = int(elevator["floor"])
                actions.append(self._nearest_floor(current_floor, load_dests))
                continue

            # Avoid changing a moving elevator's committed target every second.
            if elevator.get("state") == "moving" and elevator.get("target_floor") is not None:
                actions.append(int(elevator["target_floor"]))
                continue

            if not floors_with_waiting and elevator.get("target_floor") is None:
                actions.append(None)
                continue

            obs = self.extract_agent_observation(state, agent_id, self.num_floors)
            if self.epsilon > 0 and self.rng.rand() < self.epsilon:
                action = int(self.rng.randint(0, self.num_actions))
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.policy_net(obs_tensor).cpu().numpy()[0]
                action = int(np.argmax(q_values))

            # If the model selects an irrelevant empty floor, redirect to the
            # best currently waiting floor. This keeps evaluation robust before
            # a checkpoint has fully converged.
            if floors_with_waiting and waiting_by_floor[action] == 0:
                current_floor = int(elevator["floor"])
                action = self._best_waiting_floor(state, current_floor, excluded=actions)

            actions.append(action)

        return actions

    @staticmethod
    def extract_agent_observation(state: Dict, agent_id: int, num_floors: int) -> np.ndarray:
        elevator = state["elevators"][agent_id]
        floor_denom = max(1, num_floors - 1)
        current_floor = int(elevator["floor"])
        target = elevator.get("target_floor")
        waiting_by_floor = state.get("waiting_count_by_floor", [0] * num_floors)
        waiting_total = float(sum(waiting_by_floor))
        max_waiting = max(max(waiting_by_floor), 1)

        local = [
            current_floor / floor_denom,
            (int(elevator.get("direction", 0)) + 1) / 2,
            float(elevator.get("load", 0)) / max(1.0, float(elevator.get("capacity", 1))),
            1.0 if elevator.get("state") == "moving" else 0.0,
            1.0 if elevator.get("state") in ("door_opening", "loading") else 0.0,
            1.0 if target is not None else 0.0,
            (float(target) / floor_denom) if target is not None else 0.0,
        ]

        waiting_features = [cnt / max_waiting for cnt in waiting_by_floor]
        avg_wait = float(state.get("avg_wait_time", 0.0))
        global_features = [
            min(waiting_total / 50.0, 1.0),
            min(avg_wait / 300.0, 1.0),
            min(float(state.get("time", 0.0)) / 3600.0, 1.0),
        ]

        return np.array(local + waiting_features + global_features, dtype=np.float32)

    def _heuristic_actions(self, state: Dict) -> List[Optional[int]]:
        actions: List[Optional[int]] = []
        assigned_pickups: List[int] = []

        for i in range(self.num_elevators):
            elevator = state["elevators"][i]
            current_floor = int(elevator["floor"])
            load_dests = elevator.get("load_dests", [])

            if load_dests:
                actions.append(self._nearest_floor(current_floor, load_dests))
                continue

            if elevator.get("state") == "moving" and elevator.get("target_floor") is not None:
                actions.append(int(elevator["target_floor"]))
                continue

            target = self._best_waiting_floor(state, current_floor, excluded=assigned_pickups)
            actions.append(target)
            if target is not None:
                assigned_pickups.append(target)

        return actions

    def _best_waiting_floor(
        self,
        state: Dict,
        current_floor: int,
        excluded: Optional[Sequence[Optional[int]]] = None,
    ) -> Optional[int]:
        waiting = state.get("waiting", [])
        if not waiting:
            return None

        excluded_set = {int(f) for f in excluded or [] if f is not None}
        current_time = float(state.get("time", 0.0))
        scores: Dict[int, float] = {}

        for origin, _dest, wait_start in waiting:
            origin = int(origin)
            wait_time = max(0.0, current_time - float(wait_start))
            distance = abs(current_floor - origin)
            duplicate_penalty = 20.0 if origin in excluded_set else 0.0
            scores[origin] = scores.get(origin, 0.0) + 1.0 + 0.02 * wait_time - 0.15 * distance - duplicate_penalty

        if not scores:
            return None
        return max(scores, key=scores.get)

    @staticmethod
    def _nearest_floor(current_floor: int, floors: Sequence[int]) -> int:
        return int(min(floors, key=lambda f: abs(int(f) - current_floor)))
