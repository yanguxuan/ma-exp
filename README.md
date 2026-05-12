# 电梯调度算法实验平台

基于**早高峰真实数据**的电梯调度模拟与算法对比实验平台。
支持多电梯扩展，内置 ETA-based 工业基线算法，并预留**强化学习 / 模仿学习**接口方便扩展。

---

## 快速开始

```bash
# 1. 安装依赖
uv add numpy matplotlib

# 2. 运行实验（ETA-based 基线）
uv run python -m src.experiment_runner

# 3. 查看结果
ls results/        # 对比图表（PNG）
```

---

## 项目结构

```
ma-exp/
├── src/
│   ├── simulator.py           # 电梯模拟器（基于真实数据）
│   ├── algorithms/
│   │   ├── base.py            # 算法基类（统一接口）
│   │   └── eta_based.py       # ETA-based 工业基线
│   └── experiment_runner.py    # 实验运行器 + 可视化
├── docs/
│   ├── papers/                # 参考资料论文（5篇）
│   ├── algorithm_docs/
│   │   └── eta_based.md       # ETA-based 算法说明
│   └── reports/               # 实验报告
├── data/
│   ├── morning_peak.csv       # 早高峰客流数据
│   └── morning_peak_times.csv # 早高峰时间数据
├── results/                   # 实验输出图表
├── pyproject.toml             # Python 项目配置
└── README.md
```

---

## 模拟器说明

基于**真实早高峰数据**构建，参数自动学习自实际电梯运行记录：

| 参数 | 值 | 来源 |
|:---|---:|:---|
| 楼层 | 9 层 | 真实建筑数据 |
| 层间运行时间 | 12.5 秒 | 1→2楼实测 12.44s |
| 开关门时间 | 28 秒 | 1楼开关门实测 |
| 电梯容量 | 15 人 | 真实电梯参数 |
| 早高峰到达率 | 0.076 人/秒 | 三趟总计 42人 / 551秒 |

**多电梯扩展**：`num_elevators` 参数控制电梯数量，到达率自动按比例放大。

---

## 添加新算法

所有算法继承 `BaseAlgorithm` 即可接入实验平台：

```python
from src.algorithms.base import BaseAlgorithm

class MyRLAlgorithm(BaseAlgorithm):
    def __init__(self, num_elevators: int, num_floors: int):
        super().__init__("My-RL", num_elevators, num_floors)

    def select_actions(self, state: dict) -> list:
        """
        核心接口：根据系统状态返回每台电梯的目标楼层
        state: 见 src/simulator.py get_state()
        return: [target_floor_0, target_floor_1, ...]
        """
        return [0] * self.num_elevators  # 你的决策逻辑

    def on_episode_end(self, stats: dict):
        """RL 算法可在此更新策略"""
        pass
```

注册并运行：

```python
from src.experiment_runner import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner(ExperimentConfig())
runner.add_algorithm(MyRLAlgorithm(num_elevators=3, num_floors=9))
runner.add_eta_based()  # 添加基线对比
runner.run_all()
runner.visualize()
runner.print_summary()
```

state 结构详见 `src/simulator.py` 中的 `get_state()` 方法。

---

## 实验指标

| 指标 | 说明 | 目标 |
|:---|---|:---:|
| AWT | 乘客从到达至上电梯的平均时间（秒） | 越小越好 |
| ART | 乘客从出发到到达目的地的平均时间（秒） | 越小越好 |
| 放弃率 | 等待超过 300 秒而放弃的乘客比例 | < 3% |
| 吞吐量 | 每分钟送达乘客数（人/分钟） | 越大越好 |

---

## 参考资料

论文资料位于 `docs/papers/`：

1. Crites & Barto (1995), *Improving Elevator Performance Using RL*, NeurIPS
2. Tan (1993), *Multi-Agent RL: Independent vs Cooperative Agents*, ICML
3. Cao et al. (2022), *Deep Q Learning for Elevator Optimization*, arXiv
4. Zhang et al. (2022), *Transformer Networks for Elevator Control*, ECC
5. Vaartjes & Francois-Lavet (2025), *Novel RL for Elevator Group Control*, arXiv

---

## License

仅供学习研究使用。
