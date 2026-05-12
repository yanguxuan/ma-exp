"""
算法基类 — 统一接口
所有调度算法继承此类
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseAlgorithm(ABC):
    """所有调度算法的抽象基类"""

    def __init__(self, name: str, num_elevators: int, num_floors: int):
        self.name = name
        self.num_elevators = num_elevators
        self.num_floors = num_floors

    @abstractmethod
    def select_actions(self, state: Dict) -> List[int]:
        """
        核心接口：根据系统状态选择动作
        state: dict，由 Simulator.get_state() 返回
        return: List[int]，每台电梯的目标楼层（None=无目标/保持）
        """
        pass

    def reset(self):
        """每个 episode 开始前调用（如需清状态可重写）"""
        pass

    def on_episode_end(self, stats: Dict):
        """每个 episode 结束后调用（RL 算法可重写以更新策略）"""
        pass
