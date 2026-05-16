"""
IL+RL 推理算法：加载预训练模型，不做在线训练
"""
# =============================第一个============================
import torch
import numpy as np
from typing import List, Dict, Optional
from .base import BaseAlgorithm

# 定义神经网络结构（必须与训练时完全一致）
class PolicyNet(torch.nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_actions)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ILRLBasedAlgorithm(BaseAlgorithm):
    def __init__(self, num_elevators: int, num_floors: int, model_path: str):
        super().__init__("IL+RL", num_elevators, num_floors)
        self.num_actions = num_floors  # 目标楼层 0~num_floors-1
        self.input_dim = 4 * num_elevators + num_floors  # 与训练时一致
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNet(self.input_dim, self.num_actions).to(self.device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()
        print(f"[ILRL] 模型加载成功：{model_path}")

    def _extract_features(self, state: Dict) -> np.ndarray:
        """从状态中提取特征向量（必须与训练时的特征提取完全一致）"""
        features = []
        for e in state["elevators"]:
            # 当前楼层归一化
            floor_norm = e["floor"] / (self.num_floors - 1)
            features.append(floor_norm)
            # 方向: -1,0,1 -> 映射到 [0,1]
            dir_norm = (e["direction"] + 1) / 2
            features.append(dir_norm)
            # 负载率
            features.append(e["load"] / e["capacity"])
            # 是否处于移动状态
            features.append(1.0 if e["state"] == "moving" else 0.0)

        # 各楼层等待人数（归一化）
        waiting_by_floor = state.get("waiting_count_by_floor", [0] * self.num_floors)
        max_wait = max(waiting_by_floor) if waiting_by_floor else 1
        for cnt in waiting_by_floor:
            features.append(cnt / (max_wait + 1e-5))

        return np.array(features, dtype=np.float32)

    # def select_actions(self, state: Dict) -> List[Optional[int]]:
    #     """为每台电梯选择目标楼层"""
    #     actions = []
    #     for i in range(self.num_elevators):
    #         # 如果电梯内有乘客，直接去第一个乘客的目的地（简单规则，也可以让模型选）
    #         load_dests = state["elevators"][i].get("load_dests", [])
    #         if load_dests:
    #             actions.append(int(load_dests[0]))
    #             continue

    #         # 否则用模型预测最佳目标楼层
    #         feat = self._extract_features(state)
    #         feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
    #         with torch.no_grad():
    #             q_values = self.policy_net(feat_tensor).cpu().numpy()[0]
    #         best_action = int(np.argmax(q_values))
    #         actions.append(best_action)

    #     return actions
    def select_actions(self, state: Dict) -> List[Optional[int]]:
        actions = []
        waiting_by_floor = state.get("waiting_count_by_floor", [0]*self.num_floors)
        # 找出有等待乘客的楼层
        floors_with_waiting = [f for f, cnt in enumerate(waiting_by_floor) if cnt > 0]
        
        for i in range(self.num_elevators):
            # 电梯内有乘客：直接送第一个目的地
            load_dests = state["elevators"][i].get("load_dests", [])
            if load_dests:
                actions.append(int(load_dests[0]))
                continue
            
            # 模型预测
            feat = self._extract_features(state)
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(feat_tensor).cpu().numpy()[0]
            best_action = int(np.argmax(q_values))
            
            # 回退规则：如果预测楼层没有等待乘客，改为最多等待人数的楼层
            if waiting_by_floor[best_action] == 0 and floors_with_waiting:
                best_action = max(floors_with_waiting, key=lambda f: waiting_by_floor[f])
            
            actions.append(best_action)
        return actions
    





