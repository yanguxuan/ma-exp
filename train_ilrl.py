"""
独立的 IL+RL 训练脚本
运行方式：python train_ilrl.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.simulator import ElevatorSimulator
from src.algorithms.eta_based import ETABasedAlgorithm

# ------------------- 网络定义（与 ilrl_based.py 保持一致） -------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ------------------- 特征提取（必须与推理时一致） -------------------
def extract_features(state, num_elevators, num_floors):
    features = []
    for e in state["elevators"]:
        floor_norm = e["floor"] / (num_floors - 1)
        dir_norm = (e["direction"] + 1) / 2
        load_rate = e["load"] / e["capacity"]
        moving = 1.0 if e["state"] == "moving" else 0.0
        features.extend([floor_norm, dir_norm, load_rate, moving])
    waiting_by_floor = state.get("waiting_count_by_floor", [0] * num_floors)
    max_wait = max(waiting_by_floor) if waiting_by_floor else 1
    for cnt in waiting_by_floor:
        features.append(cnt / (max_wait + 1e-5))
    return np.array(features, dtype=np.float32)

# ------------------- 1. 收集专家数据（行为克隆） -------------------
def collect_expert_data(env, expert, num_episodes=10):
    data = []  # (state_features, action)
    for ep in range(num_episodes):
        state = env.reset()
        expert.reset()
        while env.current_time < 3600.0:
            actions = expert.select_actions(state)
            # 记录每个电梯的动作（这里简化为只记录全局状态下的动作，实际可以每个电梯单独记录）
            # 为了简化，我们记录全局特征和第一个电梯的动作（多电梯可扩展）
            feat = extract_features(state, num_elevators, num_floors)
            # 假设所有电梯动作相同，取第一个电梯的动作
            if actions[0] is not None:
                data.append((feat, actions[0]))
            state, _, done, _ = env.step(actions)
            if done:
                break
        print(f"专家 Episode {ep+1} 完成，已收集 {len(data)} 条数据")
    return data

# ------------------- 2. 行为克隆 -------------------
def behavior_cloning(net, data, epochs=10, batch_size=128, lr=1e-3):
    X = torch.tensor([d[0] for d in data], dtype=torch.float32)
    y = torch.tensor([d[1] for d in data], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = net(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"行为克隆 Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
    net.eval()

# ------------------- 3. DQN 微调 -------------------
def dqn_finetune(net, env, num_episodes=50, gamma=0.99, epsilon=0.1, batch_size=64, target_update=10):
    target_net = PolicyNet(net.fc1.in_features, net.fc3.out_features).to(device)
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    memory = deque(maxlen=10000)
    step_count = 0

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        while env.current_time < 3600.0:
            feat = extract_features(state, num_elevators, num_floors)
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = net(feat_tensor).cpu().numpy()[0]
            # ε-greedy
            if random.random() < epsilon:
                action = random.randint(0, num_floors-1)
            else:
                action = int(np.argmax(q_values))

            # 这里需要将动作转换为每个电梯的目标楼层（简单起见，所有电梯都去同一楼层）
            actions = [action] * num_elevators
            next_state, reward, done, _ = env.step(actions)

            # 存储经验
            next_feat = extract_features(next_state, num_elevators, num_floors)
            memory.append((feat, action, reward, next_feat, done))
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
                states_b = torch.tensor(np.array(states_b), dtype=torch.float32).to(device)
                actions_b = torch.tensor(actions_b, dtype=torch.long).unsqueeze(1).to(device)
                rewards_b = torch.tensor(rewards_b, dtype=torch.float32).to(device)
                next_states_b = torch.tensor(np.array(next_states_b), dtype=torch.float32).to(device)
                dones_b = torch.tensor(dones_b, dtype=torch.float32).to(device)

                q_values = net(states_b).gather(1, actions_b).squeeze()
                with torch.no_grad():
                    next_q = target_net(next_states_b).max(1)[0]
                    targets = rewards_b + gamma * next_q * (1 - dones_b)
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            step_count += 1
            if step_count % target_update == 0:
                target_net.load_state_dict(net.state_dict())

            state = next_state
            if done:
                break
        print(f"DQN Episode {ep+1}/{num_episodes} 总奖励: {total_reward:.2f}")

    return net

# ------------------- 主训练流程 -------------------
if __name__ == "__main__":
    # 参数配置（与实验时一致）
    num_floors = 9
    num_elevators = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建环境
    env = ElevatorSimulator(num_floors=num_floors, num_elevators=num_elevators, seed=42)

    # 1. 收集专家数据
    expert = ETABasedAlgorithm(num_elevators, num_floors)
    print("开始收集专家数据...")
    expert_data = collect_expert_data(env, expert, num_episodes=10)

    # 2. 行为克隆
    input_dim = 4 * num_elevators + num_floors
    net = PolicyNet(input_dim, num_floors).to(device)
    print("开始行为克隆...")
    behavior_cloning(net, expert_data, epochs=10)

    # 3. DQN 微调
    print("开始 DQN 微调...")
    net = dqn_finetune(net, env, num_episodes=30)

    # 4. 保存模型
    torch.save(net.state_dict(), "ilrl_model.pth")
    print("模型已保存为 ilrl_model.pth")