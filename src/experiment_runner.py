"""
实验运行器 + 可视化
统一接口调用模拟器和算法，预留 RL / 模仿学习接口
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime

# 动态添加 src/ 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulator import ElevatorSimulator
from src.algorithms import BaseAlgorithm, ETABasedAlgorithm
# 导入 IL+RL 算法
from src.algorithms.ilrl_based import ILRLBasedAlgorithm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================
# 实验配置
# =========================

@dataclass
class ExperimentConfig:
    num_floors: int = 9
    num_elevators: int = 3
    num_episodes: int = 5
    episode_duration: float = 3600.0  # 1小时（秒）
    seed: int = 42
    results_dir: str = "results"


# =========================
# 指标收集器
# =========================

@dataclass
class Metrics:
    algorithm_name: str
    awt: float = 0.0       # Average Waiting Time
    art: float = 0.0       # Average Ride Time
    abandon_rate: float = 0.0
    throughput: float = 0.0   # 单位时间送达乘客数
    total_served: int = 0
    total_generated: int = 0
    total_abandoned: int = 0
    episode_metrics: list = field(default_factory=list)  # 每episode详细指标


# =========================
# 实验运行器
# =========================

class ExperimentRunner:
    """
    实验运行器
    - 统一调用模拟器和各类算法
    - 预留 RL / 模仿学习接口
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.simulator: Optional[ElevatorSimulator] = None
        self.algorithms: List[BaseAlgorithm] = []
        self.results: Dict[str, Metrics] = {}
        os.makedirs(config.results_dir, exist_ok=True)

    # --------------------------
    # 注册算法（统一接口）
    # --------------------------

    def add_algorithm(self, algo: BaseAlgorithm):
        """注册算法（工业基线 / RL / IL 均使用此接口）"""
        self.algorithms.append(algo)
        print(f"✅ 已注册算法：{algo.name}")

    def add_eta_based(self):
        """添加 ETA-based 工业基线算法（快捷方法）"""
        algo = ETABasedAlgorithm(
            num_elevators=self.config.num_elevators,
            num_floors=self.config.num_floors,
        )
        self.add_algorithm(algo)

    # ---- 预留接口：RL 算法 ----
    def add_rl_algorithm(self, algo: BaseAlgorithm):
        """
        后续添加强化学习算法
        要求：algo 继承自 BaseAlgorithm
        """
        print(f"🤖 添加 RL 算法：{algo.name}")
        self.add_algorithm(algo)

    # ---- 预留接口：模仿学习 ----
    def add_imitation_algorithm(self, algo: BaseAlgorithm):
        """
        后续添加模仿学习算法
        要求：algo 继承自 BaseAlgorithm
        """
        print(f"🎓 添加模仿学习算法：{algo.name}")
        self.add_algorithm(algo)

    # --------------------------
    # 运行实验
    # --------------------------

    def run_all(self) -> Dict[str, Metrics]:
        """运行所有已注册算法"""
        for algo in self.algorithms:
            print(f"\n{'='*60}")
            print(f"开始评估算法：{algo.name}")
            print(f"{'='*60}")
            metrics = self._run_algorithm(algo)
            self.results[algo.name] = metrics
        return self.results

    def _run_algorithm(self, algo: BaseAlgorithm) -> Metrics:
        """运行单个算法，返回聚合指标"""
        metrics = Metrics(algorithm_name=algo.name)
        episode_metrics = []

        for ep in range(self.config.num_episodes):
            print(f"  Episode {ep+1}/{self.config.num_episodes}...", end=" ", flush=True)
            ep_metric = self._run_episode(algo, ep)
            episode_metrics.append(ep_metric)
            print(f"完成（送达={ep_metric['served']}，放弃={ep_metric['abandoned']}）")

        # 聚合
        metrics.episode_metrics = episode_metrics
        metrics.total_served = sum(m["served"] for m in episode_metrics)
        metrics.total_generated = sum(m["generated"] for m in episode_metrics)
        metrics.total_abandoned = sum(m["abandoned"] for m in episode_metrics)

        # 平均等待时间（所有乘客）
        all_awt = []
        all_art = []
        for m in episode_metrics:
            all_awt.extend(m["wait_times"])
            all_art.extend(m["ride_times"])
        metrics.awt = np.mean(all_awt) if all_awt else 0.0
        metrics.art = np.mean(all_art) if all_art else 0.0
        metrics.abandon_rate = (
            metrics.total_abandoned / metrics.total_generated
            if metrics.total_generated > 0 else 0.0
        )
        metrics.throughput = metrics.total_served / (self.config.num_episodes * self.config.episode_duration / 60.0)  # 人/分钟

        print(f"  汇总：AWT={metrics.awt:.1f}s, ART={metrics.art:.1f}s, 放弃率={metrics.abandon_rate:.1%}")
        return metrics

    def _run_episode(self, algo: BaseAlgorithm, ep_idx: int) -> Dict:
        """运行单个 episode"""
        seed = self.config.seed + ep_idx
        self.simulator = ElevatorSimulator(
            num_floors=self.config.num_floors,
            num_elevators=self.config.num_elevators,
            seed=seed,
        )
        state = self.simulator.reset()
        algo.reset()

        wait_times = []
        ride_times = []
        generated = 0
        served = 0
        abandoned = 0

        # 按秒逐步推进
        while self.simulator.current_time < self.config.episode_duration:
            actions = algo.select_actions(state)
            state, reward, done, info = self.simulator.step(actions)

            generated = info["generated"]
            served = info["served"]
            abandoned = info["abandoned"]

            # 收集本轮结束乘客的时间
            for p in self.simulator.all_passengers:
                if p.ride_end > 0 and p.ride_end <= self.simulator.current_time:
                    if p.wait_time > 0:
                        wait_times.append(p.wait_time)
                    if p.ride_time > 0:
                        ride_times.append(p.ride_time)

            if done:
                break

        return {
            "generated": generated,
            "served": served,
            "abandoned": abandoned,
            "wait_times": wait_times,
            "ride_times": ride_times,
        }

    # --------------------------
    # 可视化
    # --------------------------

    def visualize(self, save: bool = True):
        """生成对比图表"""
        if not self.results:
            print("⚠️ 没有可可视化的结果，请先运行实验。")
            return

        names = list(self.results.keys())
        awt = [self.results[n].awt for n in names]
        art = [self.results[n].art for n in names]
        abandon = [self.results[n].abandon_rate * 100 for n in names]
        throughput = [self.results[n].throughput for n in names]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("电梯调度算法对比实验", fontsize=14, fontweight="bold")

        # 1. 平均等待时间 AWT
        ax = axes[0, 0]
        bars = ax.bar(names, awt, color="#3498db")
        ax.set_title("平均等待时间（AWT）", fontsize=11)
        ax.set_ylabel("秒")
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, awt):
            ax.annotate(f"{v:.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=9)

        # 2. 平均行程时间 ART
        ax = axes[0, 1]
        bars = ax.bar(names, art, color="#2ecc71")
        ax.set_title("平均行程时间（ART）", fontsize=11)
        ax.set_ylabel("秒")
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, art):
            ax.annotate(f"{v:.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=9)

        # 3. 放弃率
        ax = axes[1, 0]
        bars = ax.bar(names, abandon, color="#e74c3c")
        ax.set_title("放弃率（%）", fontsize=11)
        ax.set_ylabel("百分比")
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, abandon):
            ax.annotate(f"{v:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=9)

        # 4. 吞吐量
        ax = axes[1, 1]
        bars = ax.bar(names, throughput, color="#f39c12")
        ax.set_title("吞吐量（人/分钟）", fontsize=11)
        ax.set_ylabel("人/分钟")
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, throughput):
            ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=9)

        plt.tight_layout()
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.config.results_dir, f"comparison_{timestamp}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"\n📊 图表已保存：{path}")
        plt.close()

    # --------------------------
    # 打印汇总报告
    # --------------------------

    def print_summary(self):
        """打印文字版汇总报告"""
        print(f"\n{'='*70}")
        print(f"{'算法':<20} {'AWT(秒)':>12} {'ART(秒)':>12} {'放弃率':>10} {'吞吐量(人/分)':>15}")
        print("-" * 70)
        for name, m in self.results.items():
            print(f"{name:<20} {m.awt:>12.1f} {m.art:>12.1f} {m.abandon_rate:>9.1%} {m.throughput:>15.2f}")
        print(f"{'='*70}\n")


# =========================
# 主程序
# =========================

def main():
    print("="*70)
    print("电梯调度算法实验")
    print("="*70)

    config = ExperimentConfig(
        num_floors=9,
        num_elevators=3,
        num_episodes=3,   # 快速验证用3轮，正式实验可调到20+
        episode_duration=600.0,  # 10分钟（调试用），正式实验3600.0
        seed=42,
    )

    runner = ExperimentRunner(config)

    # 注册算法（统一接口）
    runner.add_eta_based()   # ETA-based 工业基线

    # 添加 IL+RL 算法（加载预训练模型）
    # 请确保 ilrl_model.pth 文件存在于当前工作目录
    try:
        runner.add_algorithm(ILRLBasedAlgorithm(
            num_elevators=config.num_elevators,
            num_floors=config.num_floors,
            model_path="F:/aimathlab/project/ma-exp-main/ilrl_model.pth"
        ))
    except FileNotFoundError:
        print("⚠️ 未找到 ilrl_model.pth，跳过 IL+RL 算法。请先运行 train_ilrl.py 训练模型。")
    except Exception as e:
        print(f"⚠️ 加载 IL+RL 算法失败：{e}")

    # 运行实验
    runner.run_all()

    # 可视化
    runner.visualize(save=True)

    # 打印报告
    runner.print_summary()

    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()




