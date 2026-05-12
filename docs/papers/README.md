# 电梯调度强化学习 — 文献目录

整理时间：2026-04-28  
信源标准：仅收录 **NeurIPS 官方 / arXiv / MIT Media Lab** 等可靠来源的开放获取 PDF

---

## 一、已下载论文（5 篇）

---

### [1] Crites & Barto (1995) — 经典奠基论文
**文件名：** `Crites_Barto_1995_NeurIPS_Elevator_RL.pdf`

| 字段 | 内容 |
|------|------|
| 标题 | Improving Elevator Performance Using Reinforcement Learning |
| 作者 | Robert H. Crites, Andrew G. Barto |
| 发表 | NeurIPS 1995（NIPS Proceedings） |
| 来源 | https://papers.neurips.cc/paper/1073 |
| 文件大小 | ~1.5 MB |

**摘要：**  
描述了将强化学习应用于电梯调度问题的早期经典实验。电梯领域结合了大型状态空间、多智能体协同、连续状态时间、随机性等挑战。文章表明，在仿真中 RL 的结果超过了当时已知的最优启发式调度算法，证明了 RL 在真实世界问题上的强大能力。

**与本项目的关联：**  
本项目的双电梯 Q-learning 思路直接沿用了本文的核心框架（多电梯独立 TD 学习）。本文是电梯 RL 领域的必读经典。

---

### [2] Tan (1993) — 独立 Q-learning 理论基础
**文件名：** `Tan_1993_MultiAgent_IndependentQ.pdf`

| 字段 | 内容 |
|------|------|
| 标题 | Multi-Agent Reinforcement Learning: Independent versus Cooperative Agents |
| 作者 | Ming Tan |
| 发表 | ICML 1993 |
| 来源 | MIT Media Lab Open Access（https://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf） |
| 文件大小 | ~168 KB |

**摘要：**  
对独立学习（每个 agent 独立执行 Q-learning）与合作学习（agent 之间共享信息/策略）进行了系统对比实验。结论是：合作共享状态有助于加速学习，但独立学习在部分场景下也具有较强竞争力。

**与本项目的关联：**  
本项目中的 `IndependentQAgent` 直接对应本文的"Independent Q-Learning"方案，是理论依据之一。

---

### [3] Cao et al. (2022) — Deep Q-Learning 电梯仿真
**文件名：** `Cao_2022_DeepQLearning_Elevator.pdf`

| 字段 | 内容 |
|------|------|
| 标题 | Application of Deep Q Learning with Simulation Results for Elevator Optimization |
| 作者 | Zheng Cao, Raymond Guo, Caesar M. Tuguinay, Mark Pock, Jiayi Gao, Ziyu Wang |
| 发表 | arXiv:2210.00065（2022年9月，CC BY 4.0） |
| 来源 | https://arxiv.org/abs/2210.00065 |
| 文件大小 | ~453 KB |

**摘要：**  
构建了基于经典三峰模型生成模拟乘客数据的电梯仿真环境，首先实现启发式朴素控制器作为基线，再用 Deep Q Learning 与之对比。采用 MDP 框架建模，讨论了 EGCS（电梯群控系统）的随机性挑战。

**与本项目的关联：**  
三峰到达模型（上班、午饭、下班）与本项目的正弦波 lambda 到达模式动机相同；DQN 可作为本项目从 tabular Q 升级到 DQN 的参考路径。

---

### [4] Zhang et al. (2022) — Transformer 预测型调度
**文件名：** `Zhang_2022_Transformer_Elevator.pdf`

| 字段 | 内容 |
|------|------|
| 标题 | Transformer Networks for Predictive Group Elevator Control |
| 作者 | Jing Zhang, Athanasios Tsiligkaridis, Hiroshi Taguchi, Arvind Raghunathan, Daniel Nikovski |
| 发表 | European Control Conference (ECC) 2022；arXiv:2208.08948 |
| 来源 | https://arxiv.org/abs/2208.08948 |
| 文件大小 | ~1.9 MB |

**摘要：**  
提出用 Transformer 预测乘客到达目的地的概率，并结合线性回归预测剩余行程时间，实现预测型调度。在下午下班高峰中，轻流量场景 AWT（平均等待时间）最高降低 50%，中等流量降低约 15%。

**与本项目的关联：**  
代表了从反应式（reactive）调度到预测式（predictive）调度的演进方向，可作为本项目的未来工作方向参考。

---

### [5] Vaartjes & Francois-Lavet (2025) — 最新 RL EGCS
**文件名：** `Vaartjes_2025_Novel_RL_EGCS.pdf`

| 字段 | 内容 |
|------|------|
| 标题 | Novel RL approach for efficient Elevator Group Control Systems |
| 作者 | Nathan Vaartjes, Vincent Francois-Lavet |
| 发表 | arXiv:2507.00011（2025年6月） |
| 来源 | https://arxiv.org/abs/2507.00011 |
| 文件大小 | ~2.4 MB |

**摘要：**  
以阿姆斯特丹自由大学的真实6电梯15层建筑为场景，将 EGCS 建模为 MDP，训练端到端 RL 智能体（Dueling Double Deep Q-learning）。关键创新：新颖的动作空间编码解决组合爆炸问题、引入"infra-steps"模拟连续乘客到达、定制奖励信号提升学习效率。在实验中超越传统规则控制器。

**与本项目的关联：**  
是目前电梯 RL 领域最新研究（2025），其 infra-step 设计思想与本项目的"每步末尾 spawn" 机制类似；D3QN 架构是 tabular Q 的直接升级路径。

---

## 二、未下载但推荐的重要文献

以下论文因版权限制无法直接提供 PDF，但可通过学校图书馆或 IEEEXplore/Springer 访问：

| 编号 | 论文 | 来源 | 重要性 |
|------|------|------|--------|
| [A] | Crites & Barto (1998), "Elevator Group Control Using Multiple RL Agents", Machine Learning | Springer | ★★★★★ 多电梯 MARL 的后续深化 |
| [B] | Liu et al. (2013), "Dispatching algorithm design for elevator group control system with Q-learning", ICCA | IEEE | ★★★★ Q-learning 电梯调度的代表工程论文 |
| [C] | Advanced Engineering Informatics (2024), "Traffic pattern-aware elevator dispatching via deep RL" | ScienceDirect | ★★★★ SMDP + 新颖状态表示的 SOTA 方案 |

---

## 三、文件目录

```
papers/
├── README.md                                   <- 本文档（文献目录与摘要）
├── Crites_Barto_1995_NeurIPS_Elevator_RL.pdf   <- [1] NeurIPS 1995 经典
├── Tan_1993_MultiAgent_IndependentQ.pdf         <- [2] IQL 理论基础
├── Cao_2022_DeepQLearning_Elevator.pdf          <- [3] arXiv 2022 DQN
├── Zhang_2022_Transformer_Elevator.pdf          <- [4] arXiv 2022 Transformer
└── Vaartjes_2025_Novel_RL_EGCS.pdf             <- [5] arXiv 2025 最新
```

---

## 四、阅读推荐顺序

1. **Tan 1993** → 理解独立 Q-learning 的数学原理
2. **Crites & Barto 1995** → 了解电梯 RL 的经典问题定义
3. **Cao 2022** → 看看现代 DQN 如何解同一问题
4. **Vaartjes 2025** → 最新 SOTA，了解前沿设计思路
5. **Zhang 2022** → 了解预测型调度的方向

---

*信源可靠性声明：所有已下载 PDF 均来自 NeurIPS 官方 Proceedings、MIT Media Lab、arXiv.org，均为开放获取版本，无版权风险。*
