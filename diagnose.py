"""
快速诊断 — 分析放弃原因
"""
import sys
sys.path.insert(0, '.')

from src.simulator import ElevatorSimulator
from src.algorithms.eta_based import ETABasedAlgorithm
import numpy as np

sim = ElevatorSimulator(num_floors=9, num_elevators=1, seed=42)
state = sim.reset()
alg = ETABasedAlgorithm(num_elevators=1, num_floors=9)

# 统计
served_times = []
abandon_times = []
waiting_counts = []
elevator_log = []

for t in range(600):  # 前600秒
    actions = alg.select_actions(state)
    state, reward, done, info = sim.step(actions)
    
    e = state['elevators'][0]
    elevator_log.append((state['time'], e['floor'], e['state'], e['load'], e['target_floor'], len(state['waiting'])))
    
    if t % 50 == 0 or t < 10:
        print(f"T={state['time']:5.0f}s | elev=floor{e['floor']} {e['state']:15s} load={e['load']:2d} target={e['target_floor']} | waiting={len(state['waiting']):3d} | served={info['served']:2d} abandoned={info['abandoned']:2d}")

print(f"\n===== 600秒统计 =====")
print(f"送达: {info['served']}")
print(f"放弃: {info['abandoned']}")
print(f"生成: {info['generated']}")
print(f"上车: {info['boarded']}")
print(f"平均等待: {info['avg_wait']:.1f}秒")

# 计算放弃率
total = info['served'] + info['abandoned']
print(f"\n放弃率: {info['abandoned']/(info['abandoned']+info['boarded'])*100:.1f}%")

# 分析: 电梯在空载时 target_floor 是什么
print("\n===== 电梯空载时的目标分析 =====")
empty_targets = {}
for t, floor, st, load, target, waiting in elevator_log:
    if load == 0:
        key = (st, target)
        empty_targets[key] = empty_targets.get(key, 0) + 1
for (st, target), count in sorted(empty_targets.items(), key=lambda x: -x[1]):
    print(f"  状态={st:15s} target={target} => {count}步")

print("\n===== 电梯有载时的目标分析 =====")
loaded_targets = {}
for t, floor, st, load, target, waiting in elevator_log:
    if load > 0:
        key = target
        loaded_targets[key] = loaded_targets.get(key, 0) + 1
for target, count in sorted(loaded_targets.items(), key=lambda x: -x[1]):
    print(f"  target={target} => {count}步 ({count/6:.1f}秒)")
