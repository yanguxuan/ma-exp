"""
调试脚本 - 检查模拟器和算法
"""
import sys
sys.path.insert(0, '.')

from src.simulator import ElevatorSimulator
from src.algorithms.eta_based import ETABasedAlgorithm

# 创建模拟器（1台电梯，运行60秒）
sim = ElevatorSimulator(num_floors=9, num_elevators=1, seed=42)
state = sim.reset()

alg = ETABasedAlgorithm(num_elevators=1, num_floors=9)

print("=== 初始状态 ===")
print(f"时间: {state['time']}秒")
print(f"等待乘客数: {len(state['waiting'])}")
print(f"电梯状态: {state['elevators'][0]}")

# 运行150个时间步（足够电梯从0楼到6楼并下客）
for t in range(150):
    # 算法选择动作（传入整个state，返回actions列表）
    actions = alg.select_actions(state)
    
    e_floor = state['elevators'][0]['floor']
    e_state = state['elevators'][0]['state']
    e_target = state['elevators'][0]['target_floor']
    waiting_count = len(state['waiting'])
    
    print(f"t={t+1:2d} (T={state['time']:5.1f}s): action={actions} | floor={e_floor} state={e_state:16s} target={e_target} | waiting={waiting_count}")
    
    # 执行一步
    state, reward, done, info = sim.step(actions)
    
    if done:
        break

print("\n=== 10步后状态 ===")
print(f"总生成: {info['generated']}")
print(f"总上车: {info['boarded']}")
print(f"总送达: {info['served']}")
print(f"总放弃: {info['abandoned']}")
