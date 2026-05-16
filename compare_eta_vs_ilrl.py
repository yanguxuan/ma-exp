"""
对比实验：ETA-based 基线 vs IL+RL 算法
"""
import sys, os, numpy as np

os.makedirs("results", exist_ok=True)

from src.simulator import ElevatorSimulator
from src.algorithms.eta_based import ETABasedAlgorithm
from src.algorithms.ilrl_based import ILRLBasedAlgorithm
from src.experiment_runner import Metrics, ExperimentConfig

# 参数
NUM_FLOORS = 9
NUM_ELEVATORS = 3
NUM_EPISODES = 3  # 每算法跑3轮
DURATION = 3600.0  # 1小时
SEED = 42

algorithms = [
    ("ETA-based", ETABasedAlgorithm(NUM_ELEVATORS, NUM_FLOORS)),
    ("IL+RL", ILRLBasedAlgorithm(NUM_ELEVATORS, NUM_FLOORS, "ilrl_model.pth")),
]

all_metrics = {}

for name, algo in algorithms:
    print(f"\n{'='*60}")
    print(f"评估算法：{name}")
    print(f"{'='*60}")

    ep_metrics = []
    for ep in range(NUM_EPISODES):
        env = ElevatorSimulator(num_floors=NUM_FLOORS, num_elevators=NUM_ELEVATORS, seed=SEED + ep)
        state = env.reset()
        algo.reset()

        wait_times, ride_times = [], []
        while env.current_time < DURATION:
            actions = algo.select_actions(state)
            state, reward, done, info = env.step(actions)
            if done:
                break

        # 收集统计
        for p in env.all_passengers:
            if p.wait_end > 0 and p.wait_time > 0:
                wait_times.append(p.wait_time)
            if p.ride_end > 0 and p.ride_time > 0:
                ride_times.append(p.ride_time)

        ep_metrics.append(info)
        print(f"  Episode {ep+1}: 送达={info['served']} 放弃={info['abandoned']} 上车={info['boarded']}")

    # 聚合
    avg_served = np.mean([m["served"] for m in ep_metrics])
    avg_abandoned = np.mean([m["abandoned"] for m in ep_metrics])
    avg_generated = np.mean([m["generated"] for m in ep_metrics])
    avg_boarded = np.mean([m["boarded"] for m in ep_metrics])
    abandon_rate = avg_abandoned / (avg_abandoned + avg_boarded) if (avg_abandoned + avg_boarded) > 0 else 0
    avg_wait = np.mean(wait_times) if wait_times else 0
    avg_ride = np.mean(ride_times) if ride_times else 0

    all_metrics[name] = {
        "served": avg_served,
        "abandoned": avg_abandoned,
        "generated": avg_generated,
        "boarded": avg_boarded,
        "abandon_rate": abandon_rate,
        "avg_wait": avg_wait,
        "avg_ride": avg_ride,
    }

    print(f"  汇总: 平均送达={avg_served:.1f} 放弃={avg_abandoned:.1f} AWT={avg_wait:.1f}s ART={avg_ride:.1f}s")

# ========== 对比结果 ==========
print("\n\n" + "="*70)
print(f"{'算法':<20} {'送达':>8} {'放弃':>8} {'AWT(秒)':>10} {'ART(秒)':>10} {'放弃率':>10}")
print("-"*70)
for name, m in all_metrics.items():
    print(f"{name:<20} {m['served']:>8.1f} {m['abandoned']:>8.1f} {m['avg_wait']:>10.1f} {m['avg_ride']:>10.1f} {m['abandon_rate']:>9.1%}")
print("="*70)

# ========== 可视化 ==========
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = list(all_metrics.keys())
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("ETA-based vs IL+RL 算法对比（3台电梯）", fontsize=14, fontweight="bold")

colors = ["#3498db", "#e74c3c"]

metrics_plot = [
    ("平均送达人数", [all_metrics[n]["served"] for n in names], "#2ecc71"),
    ("平均放弃人数", [all_metrics[n]["abandoned"] for n in names], "#e74c3c"),
    ("平均等待时间(AWT秒)", [all_metrics[n]["avg_wait"] for n in names], "#3498db"),
    ("平均行程时间(ART秒)", [all_metrics[n]["avg_ride"] for n in names], "#f39c12"),
]

for ax, (title, vals, color) in zip(axes.flat, metrics_plot):
    bars = ax.bar(names, vals, color=color, width=0.4)
    ax.set_title(title, fontsize=11)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

plt.tight_layout()
path = "results/eta_vs_ilrl_comparison.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n📊 对比图已保存：{path}")
plt.close()
