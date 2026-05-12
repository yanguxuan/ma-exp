"""
电梯模拟器 — 基于早高峰真实数据
支持可扩展电梯数，时间单位为真实秒数
"""

import numpy as np
import random
import csv
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


# =========================
# 数据类
# =========================

@dataclass
class Passenger:
    origin: int
    dest: int
    wait_start: float = 0.0   # 开始等待的时间（秒）
    wait_end: float = 0.0     # 上电梯的时间
    ride_end: float = 0.0     # 到达目的地的时间

    @property
    def wait_time(self) -> float:
        return self.wait_end - self.wait_start

    @property
    def ride_time(self) -> float:
        return self.ride_end - self.wait_end


@dataclass
class ElevatorState:
    id: int
    floor: float = 0.0          # 当前楼层（可以是小数，表示运行中）
    direction: int = 0          # -1=下行, 0=静止, 1=上行
    load: List[Passenger] = field(default_factory=list)
    state: str = "idle"         # idle/moving/door_opening/loading
    state_timer: float = 0.0    # 当前状态剩余时间（秒）
    target_floor: Optional[int] = None


# =========================
# 模拟器核心类
# =========================

class ElevatorSimulator:
    """
    基于早高峰真实数据的电梯模拟器
    - 时间单位：真实秒数
    - 支持 1~N 台电梯
    - 从真实数据自动学习参数
    """

    def __init__(
        self,
        num_floors: int = 9,
        num_elevators: int = 1,
        data_path: Optional[str] = None,
        seed: int = 0,
    ):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.seed = seed
        self.rng = random.Random(seed)
        np_rng_seed = seed
        self.np_rng = np.random.RandomState(np_rng_seed)

        # 从数据学习参数，或使用默认值
        self.params = self._learn_params_from_data(data_path)

        # 电梯列表
        self.elevators: List[ElevatorState] = [
            ElevatorState(id=i) for i in range(num_elevators)
        ]

        # 全局状态
        self.current_time: float = 0.0
        self.waiting: List[Passenger] = []
        self.generated = 0
        self.boarded = 0
        self.served = 0
        self.abandoned = 0
        self.all_passengers: List[Passenger] = []  # 所有乘客（用于统计）
        self.max_wait_time = 300.0  # 最大等待时间（秒），超过则放弃

        # 早高峰客流模式（基于真实数据）
        self.arrival_rate_per_elevator = self._compute_arrival_rates()

        # 扩大客流：N 台电梯 = N 倍流量
        for f in list(self.arrival_rate_per_elevator.keys()):
            self.arrival_rate_per_elevator[f] *= num_elevators

    def _learn_params_from_data(self, data_path: Optional[str]) -> Dict:
        if data_path:
            pass  # 从CSV文件读取（暂时留接口）

        # 从用户提供的表格计算默认值
        params = {
            "floor_travel_time": 12.5,
            "door_open_time": 14.0,
            "door_close_time": 14.0,
            "passenger_time": 2.0,
            "capacity": 15,
            # 早高峰到达率：42人分3趟，总耗时=199+169+183=551秒
            # 正确值 = 42/551 ≈ 0.076人/秒（之前误用184秒只算了第一趟）
            "arrival_rate_1f": 42.0 / (199.53 + 169.07 + 183.37),
        }
        return params

    def _compute_arrival_rates(self) -> Dict[int, float]:
        rates = defaultdict(float)
        rates[0] = self.params["arrival_rate_1f"]  # 1楼（索引0）
        return rates

    def reset(self) -> Dict:
        self.current_time = 0.0
        self.waiting.clear()
        self.generated = 0
        self.boarded = 0
        self.served = 0
        self.abandoned = 0
        self.all_passengers.clear()

        for e in self.elevators:
            e.floor = 0.0
            e.direction = 0
            e.load.clear()
            e.state = "idle"
            e.state_timer = 0.0
            e.target_floor = None

        # 初始生成一批乘客
        self._spawn_passengers(n=int(self.params["arrival_rate_1f"] * 60))
        return self.get_state()

    def _spawn_passengers(self, n: int):
        """生成 n 个新乘客（基于早高峰模式）"""
        # 目的地权重：基于真实数据三趟下客分布
        # 列表索引 0=1楼, 1=2楼, ..., 8=9楼
        raw = [0, 0, 2, 5, 2, 5, 15, 12, 9]
        # 截断或补零以匹配楼层数
        if len(raw) > self.num_floors:
            raw = raw[:self.num_floors]
        elif len(raw) < self.num_floors:
            raw = raw + [0] * (self.num_floors - len(raw))
        total = sum(raw)

        dest_probs = [0.0] * self.num_floors
        if total > 0:
            for f in range(self.num_floors):
                dest_probs[f] = raw[f] / total

        for _ in range(n):
            origin = 0  # 早高峰：所有人从1楼上电梯
            if total > 0:
                dest = self.np_rng.choice(range(self.num_floors), p=dest_probs)
            else:
                dest = self.rng.randint(1, self.num_floors - 1)

            p = Passenger(origin=origin, dest=dest, wait_start=self.current_time)
            self.waiting.append(p)
            self.generated += 1
            self.all_passengers.append(p)

    def get_state(self) -> Dict:
        return {
            "time": self.current_time,
            "elevators": [
                {
                    "id": e.id,
                    "floor": int(e.floor),
                    "direction": e.direction,
                    "load": len(e.load),
                    "load_dests": [p.dest for p in e.load],  # 乘客目的地列表
                    "capacity": self.params["capacity"],
                    "state": e.state,
                    "target_floor": e.target_floor,
                }
                for e in self.elevators
            ],
            "waiting": [(p.origin, p.dest, p.wait_start) for p in self.waiting],
            "waiting_count_by_floor": self._count_waiting_by_floor(),
            "avg_wait_time": self._avg_wait_time(),
        }

    def _count_waiting_by_floor(self) -> List[int]:
        counts = [0] * self.num_floors
        for p in self.waiting:
            counts[p.origin] += 1
        return counts

    def _avg_wait_time(self) -> float:
        if not self.waiting:
            return 0.0
        return (self.current_time - sum(p.wait_start for p in self.waiting)) / len(self.waiting)

    def step(self, actions: List[Optional[int]]) -> Tuple[Dict, float, bool, Dict]:
        self.current_time += 1.0
        reward = 0.0

        for i, e in enumerate(self.elevators):
            if actions[i] is not None:
                e.target_floor = actions[i]
            self._update_elevator(e)
            reward += self._compute_elevator_reward(e)

        # 生成新乘客
        lambda_per_sec = self.params["arrival_rate_1f"]
        n_new = self.np_rng.poisson(lambda_per_sec)
        if n_new > 0:
            self._spawn_passengers(n_new)

        # 放弃等待的乘客
        still_waiting = []
        for p in self.waiting:
            wait_duration = self.current_time - p.wait_start
            if wait_duration > self.max_wait_time:
                self.abandoned += 1
                reward -= 50.0
            else:
                still_waiting.append(p)
        self.waiting = still_waiting

        info = {
            "time": self.current_time,
            "generated": self.generated,
            "boarded": self.boarded,
            "served": self.served,
            "abandoned": self.abandoned,
            "avg_wait": self._avg_wait_time(),
        }

        done = self.current_time >= 3600.0  # 1小时
        state = self.get_state()
        return state, reward, done, info

    def _update_elevator(self, e: ElevatorState):
        cap = self.params["capacity"]

        if e.state == "idle":
            if e.target_floor is not None:
                if e.target_floor == int(e.floor):
                    # 已到达目标楼层，直接开门
                    e.state = "door_opening"
                    e.state_timer = self.params["door_open_time"]
                elif e.target_floor != int(e.floor):
                    e.direction = 1 if e.target_floor > int(e.floor) else -1
                    e.state = "moving"
                    e.state_timer = self.params["floor_travel_time"] * abs(e.target_floor - int(e.floor))

        elif e.state == "moving":
            e.state_timer -= 1.0
            if e.state_timer > 0:
                pass  # 正在运行
            elif e.state_timer <= 0:
                if e.target_floor is not None:
                    e.floor = float(e.target_floor)
                else:
                    e.floor = round(e.floor)
                e.state = "door_opening"
                e.state_timer = self.params["door_open_time"]

        elif e.state == "door_opening":
            e.state_timer -= 1.0
            if e.state_timer <= 0:
                e.state = "loading"
                self._handle_alighting(e)
                self._handle_boarding(e)
                e.state_timer = self.params["door_close_time"]

        elif e.state == "loading":
            e.state_timer -= 1.0
            if e.state_timer <= 0:
                e.state = "idle"
                e.target_floor = None  # 清除目标，让算法重新计算

    def _handle_alighting(self, e: ElevatorState):
        remain = []
        for p in e.load:
            if p.dest == int(e.floor):
                p.ride_end = self.current_time
                self.served += 1
            else:
                remain.append(p)
        e.load = remain

    def _handle_boarding(self, e: ElevatorState):
        cap = self.params["capacity"]
        free = cap - len(e.load)
        if free <= 0:
            return

        here = sorted(
            [p for p in self.waiting if p.origin == int(e.floor)],
            key=lambda p: -p.wait_start
        )

        for p in here[:free]:
            p.wait_end = self.current_time
            e.load.append(p)
            self.waiting.remove(p)
            self.boarded += 1

    def _compute_elevator_reward(self, e: ElevatorState) -> float:
        reward = 0.0
        for p in e.load:
            if p.dest == int(e.floor):
                reward += 10.0
        return reward
