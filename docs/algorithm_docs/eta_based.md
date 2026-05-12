# ETA-based 电梯调度算法说明

## 一、算法原理

### 1.1 核心思想

**ETA（Estimated Time of Arrival，预计到达时间）** 算法是现代电梯群控系统（EGCS）的核心调度策略。

核心逻辑：
> 对每一台电梯，计算它到达每一个候梯乘客所在楼层的**预计时间**，选择 ETA 最小的乘客所在的楼层作为该电梯的下一个目标。

### 1.2 ETA 计算公式

```
ETA(i, p) = T_travel(i→p.origin)    // 运行时间
           + T_door(i)                  // 开关门时间
           + P_direction(i, p)          // 方向惩罚
           + P_load(i)                   // 负载惩罚
           - B_wait(p)                  // 等待奖励（等待久的优先）
```

| 因子 | 说明 | 典型值 |
|:---|:---|---:|
| T_travel | 电梯到乘客楼层的运行时间 | `distance × 12.5s` |
| T_door | 当前开关门剩余时间 | `0 ~ 28s` |
| P_direction | 方向与乘客位置不一致时的惩罚 | `0 or 20s` |
| P_load | 电梯负载高时的惩罚 | `load_ratio × 15s` |
| B_wait | 乘客已等待时间的奖励（优先服务等久的人） | `wait_time × 0.1` |

### 1.3 决策流程

```
对每台电梯 i：
  对每位等待乘客 p：
    计算 ETA(i, p)
  选择 ETA 最小的乘客 p*
  设置电梯 i 的目标楼层 = p*.origin
```

---

## 二、工业应用证据

### 2.1 Otis Compass®

Otis 是全球最大的电梯制造商之一，其 **Compass® 目的地调度系统** 核心就是 ETA 类算法。

> "The dispatching decision is based on the estimated time of arrival at each hall call."
> — Otis Compass® Technical Documentation

**应用特点**：
- 实时计算每台电梯到每个候梯请求的 ETA
- 综合考虑距离、方向、负载、等待时间
- 动态调整分配策略

### 2.2 KONE Destination Control

KONE（通力）的 **Destination Control System** 同样使用 ETA 作为核心指标。

**应用特点**：
- 目的地预输入（乘客在候梯前选择目的地）
- 系统计算多台电梯的 ETA，最优分配
- 显著减少平均等待时间（官方数据：减少 30-50%）

### 2.3 学术研究支持

| 论文 | 发表 | 结论 |
|---|---|---|
| Crites & Barto 1995, NeurIPS | 1995 | ETA类调度在仿真中超越当时最优启发式算法 |
| Cao et al. 2022, arXiv | 2022 | DQN + ETA 特征工程，效果显著优于规则方法 |
| Vaartjes & Francois-Lavet 2025, arXiv | 2025 | 现代RL方法的核心状态表示仍基于 ETA |

---

## 三、与工业界其他算法对比

| 算法 | 工业应用 | 优点 | 缺点 |
|---|---|---|---|
| **ETA-based** | ✅ Otis, KONE 主流使用 | 综合考虑多因素，效果稳定 | 计算量中等 |
| SCAN/LOOK | ⚠️ 早期系统，现已淘汰 | 实现极简单 | 忽略等待时间，体验差 |
| 模糊逻辑 | ✅ 三菱、日立部分型号 | 可建模人类经验 | 规则设计困难，需专家知识 |
| 目的地控制（DCS） | ✅ 高端新装电梯 | 效率最高 | 需要改造候梯厅硬件 |

---

## 四、本实现的关键设计

### 4.1 代码位置

```
src/algorithms/eta_based.py   ← ETA-based 算法实现
src/simulator.py            ← 模拟器（基于真实早高峰数据）
```

### 4.2 接口设计

```python
class ETABasedAlgorithm(BaseAlgorithm):
    def select_actions(self, state: Dict) -> List[Optional[int]]:
        """
        输入：系统状态 state（由模拟器提供）
        输出：每台电梯的目标楼层
        """
```

### 4.3 ETA 计算实现（简化版）

```python
for p in waiting_passengers:
    distance = abs(elevator.floor - p.origin)
    travel_time = distance * FLOOR_TRAVEL_TIME

    # 方向惩罚
    if (elevator.direction > 0 and p.origin < elevator.floor):
        direction_penalty = 20.0
    else:
        direction_penalty = 0.0

    # 负载惩罚
    load_penalty = (elevator.load / CAPACITY) * 15.0

    # 等待奖励（等越久越优先）
    wait_bonus = p.wait_time * 0.1

    eta = travel_time + direction_penalty + load_penalty - wait_bonus
```

### 4.4 参数说明

| 参数 | 值 | 来源 |
|---|---|---|
| 层间运行时间 | 12.5 秒 | 从真实数据计算（1→2楼：12.44s） |
| 开关门时间 | 14.0 秒 | 从真实数据推算（1楼开关门共约28s） |
| 电梯容量 | 15 人 | 真实电梯参数 |
| 方向惩罚 | 20.0 秒 | 工程经验值（掉头额外时间） |

---

## 五、实验评估

### 5.1 评估指标

| 指标 | 说明 | 目标 |
|---|---|---|
| 平均等待时间（AWT） | 乘客从按按钮到上电梯的时间 | 越小越好 |
| 平均行程时间（ART） | 乘客从出发到到达目的地的时间 | 越小越好 |
| 放弃率 | 等待超时放弃的比例 | 越小越好 |
| 服务效率 | 成功送达 / 总生成 | 越大越好 |

### 5.2 与 RL 算法对比（后续实验）

ETA-based 作为**工业基线**，后续将对比：
- Independent Q-Learning
- Centralized Q-Learning  
- Deep Q-Network（DQN）
- 模仿学习（Imitation Learning）

---

## 六、参考文献

1. Crites, R. H., & Barto, A. G. (1995). *Improving Elevator Performance Using Reinforcement Learning*. NeurIPS 1995.
2. Otis Elevator Company. *Compass® Destination Dispatch System Technical Guide*.
3. KONE Corporation. *Destination Control System White Paper*.
4. Cao, Z. et al. (2022). *Application of Deep Q Learning for Elevator Optimization*. arXiv:2210.00065.
5. Vaartjes, N., & Francois-Lavet, V. (2025). *Novel RL approach for efficient Elevator Group Control Systems*. arXiv:2507.00011.

---

*文档版本：v1.0 | 更新时间：2026-05-12*
