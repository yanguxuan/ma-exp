"""
Eta-based 调度算法
工业应用：Otis Compass®, KONE Destination Control
"""

from typing import List, Dict, Optional
from .base import BaseAlgorithm


class ETABasedAlgorithm(BaseAlgorithm):
    """
    ETA-based（预计到达时间）调度算法

    工业证据：
    - Otis Compass® 系统：基于 ETA 的电梯群控
    - KONE Destination Control：使用预计到达时间优化分配
    - 学术论文：Crites & Barto 1995（NeurIPS）验证了 ETA 类方法的有效性

    原理：
    为每台电梯计算到每个等待乘客的预计到达时间（ETA），
    选择 ETA 最小（最快到达）的乘客所在楼层作为目标。
    """

    def __init__(self, num_elevators: int, num_floors: int):
        super().__init__("Eta-based", num_elevators, num_floors)
        # 算法超参数
        self.door_time = 14.0          # 开关门时间（秒）
        self.floor_travel_time = 12.5 # 层间运行时间（秒）
        self.wait_weight = 0.1          # 等待时间权重（优先级）

    def select_actions(self, state: Dict) -> List[Optional[int]]:
        """
        为每台电梯选择目标楼层
        return: [目标楼层或 None（无目标时）]
        """
        actions: List[Optional[int]] = []

        for i in range(self.num_elevators):
            elevator_state = state["elevators"][i]
            current_floor = elevator_state["floor"]
            current_load = elevator_state["load"]
            target_floor = elevator_state["target_floor"]

            # 如果电梯内有乘客，优先送达到目的地
            if current_load > 0:
                # 从 state 中获取乘客目的地列表
                load_dests = elevator_state.get("load_dests", [])
                if load_dests:
                    # 选第一个目的地，转为普通 int 避免 numpy 类型问题
                    dest = int(load_dests[0])
                else:
                    dest = 2  # fallback
                actions.append(dest)
                continue

            # 如果电梯空载，继续前往已有目标
            if target_floor is not None:
                actions.append(target_floor)
                continue
            
            # 电梯空载且无目标，计算所有等待乘客的 ETA
            best_target: Optional[int] = None
            best_eta = float("inf")

            for p_origin, p_dest, p_wait_start in state["waiting"]:
                origin_floor = p_origin  # p_origin 是整数楼层

                # --- ETA 计算 ---
                # 1. 距离成本
                distance = abs(current_floor - origin_floor)
                travel_time = distance * self.floor_travel_time

                # 2. 方向惩罚：反方向需要额外时间
                direction = elevator_state["direction"]
                if direction > 0 and origin_floor < current_floor:
                    direction_penalty = 20.0  # 需要掉头
                elif direction < 0 and origin_floor > current_floor:
                    direction_penalty = 20.0
                else:
                    direction_penalty = 0.0

                # 3. 负载惩罚：接近满载的电梯 ETA 偏大
                load_ratio = current_load / elevator_state["capacity"]
                load_penalty = load_ratio * 15.0

                # 4. 等待时间奖励：等待久的乘客降低其 ETA（优先服务）
                wait_duration = state["time"] - p_wait_start
                wait_bonus = wait_duration * self.wait_weight

                eta = travel_time + direction_penalty + load_penalty - wait_bonus

                if eta < best_eta:
                    best_eta = eta
                    best_target = origin_floor

            actions.append(best_target)

        return actions
