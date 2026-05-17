"""Compare ETA-based, IL+RL, and Multi-Agent RL elevator dispatching."""

from __future__ import annotations

import os

import numpy as np

from src.algorithms.eta_based import ETABasedAlgorithm
from src.algorithms.ilrl_based import ILRLBasedAlgorithm
from src.algorithms.multi_agent_rl import MultiAgentRLAlgorithm
from src.simulator import ElevatorSimulator
from src.visualization import plot_algorithm_comparison


NUM_FLOORS = 9
NUM_ELEVATORS = 3
NUM_EPISODES = 3
DURATION = 3600.0
SEED = 42


def evaluate_algorithm(name, algo):
    episode_infos = []
    all_wait_times = []
    all_ride_times = []

    print("\n" + "=" * 60)
    print(f"Evaluating algorithm: {name}")
    print("=" * 60)

    for ep in range(NUM_EPISODES):
        env = ElevatorSimulator(
            num_floors=NUM_FLOORS,
            num_elevators=NUM_ELEVATORS,
            seed=SEED + ep,
        )
        state = env.reset()
        algo.reset()
        info = {"generated": 0, "boarded": 0, "served": 0, "abandoned": 0}

        while env.current_time < DURATION:
            actions = algo.select_actions(state)
            state, _reward, done, info = env.step(actions)
            if done:
                break

        for passenger in env.all_passengers:
            if passenger.wait_end > 0 and passenger.wait_time > 0:
                all_wait_times.append(passenger.wait_time)
            if passenger.ride_end > 0 and passenger.ride_time > 0:
                all_ride_times.append(passenger.ride_time)

        episode_infos.append(info)
        print(
            f"  Episode {ep + 1}: served={info['served']} "
            f"abandoned={info['abandoned']} boarded={info['boarded']}"
        )

    avg_served = float(np.mean([m["served"] for m in episode_infos]))
    avg_abandoned = float(np.mean([m["abandoned"] for m in episode_infos]))
    avg_generated = float(np.mean([m["generated"] for m in episode_infos]))
    avg_boarded = float(np.mean([m["boarded"] for m in episode_infos]))
    abandon_rate = avg_abandoned / avg_generated if avg_generated > 0 else 0.0
    avg_wait = float(np.mean(all_wait_times)) if all_wait_times else 0.0
    avg_ride = float(np.mean(all_ride_times)) if all_ride_times else 0.0

    result = {
        "served": avg_served,
        "abandoned": avg_abandoned,
        "generated": avg_generated,
        "boarded": avg_boarded,
        "abandon_rate": abandon_rate,
        "avg_wait": avg_wait,
        "avg_ride": avg_ride,
    }

    print(
        f"  Summary: served={avg_served:.1f} abandoned={avg_abandoned:.1f} "
        f"AWT={avg_wait:.1f}s ART={avg_ride:.1f}s abandon_rate={abandon_rate:.1%}"
    )
    return result


def build_algorithms():
    algorithms = [("ETA-based", ETABasedAlgorithm(NUM_ELEVATORS, NUM_FLOORS))]

    if os.path.exists("ilrl_model.pth"):
        algorithms.append(
            ("IL+RL", ILRLBasedAlgorithm(NUM_ELEVATORS, NUM_FLOORS, "ilrl_model.pth"))
        )
    else:
        print("Skip IL+RL: ilrl_model.pth not found.")

    algorithms.append(
        (
            "Multi-Agent RL",
            MultiAgentRLAlgorithm(
                NUM_ELEVATORS,
                NUM_FLOORS,
                model_path="marl_model.pth",
                heuristic_fallback=True,
            ),
        )
    )
    return algorithms


def main():
    os.makedirs("results", exist_ok=True)
    all_metrics = {}

    for name, algo in build_algorithms():
        all_metrics[name] = evaluate_algorithm(name, algo)

    print("\n" + "=" * 78)
    print(
        f"{'Algorithm':<20} {'Served':>8} {'Abandoned':>10} "
        f"{'AWT(s)':>10} {'ART(s)':>10} {'Abandon%':>10}"
    )
    print("-" * 78)
    for name, m in all_metrics.items():
        print(
            f"{name:<20} {m['served']:>8.1f} {m['abandoned']:>10.1f} "
            f"{m['avg_wait']:>10.1f} {m['avg_ride']:>10.1f} {m['abandon_rate']:>9.1%}"
        )
    print("=" * 78)

    save_path = "results/eta_ilrl_marl_comparison.png"
    fig = plot_algorithm_comparison(all_metrics, save_path=save_path)
    fig.clear()
    print(f"Saved comparison figure to {save_path}")


if __name__ == "__main__":
    main()
