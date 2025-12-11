import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from enviornment import WumpusEnv


def state_to_tensor(state, H, W):
    """
    Convert env state -> tensor channels (C, H, W).
    Channels:
      0 : agent
      1 : gold (0 if taken / None)
      2 : exit
      3 : pits
      4 : wumpus_up
      5 : wumpus_down
      6 : wumpus_left
      7 : wumpus_right
    """
    C = 8
    ch = np.zeros((C, H, W), dtype=np.float32)

    # agent
    ar, ac = state["agent"]
    ch[0, ar, ac] = 1.0

    # gold
    if state["gold"] is not None:
        gr, gc = state["gold"]
        ch[1, gr, gc] = 1.0

    # exit
    er, ec = state["exit"]
    ch[2, er, ec] = 1.0

    # pits
    for pr, pc in state["pits"]:
        ch[3, pr, pc] = 1.0

    # wumpus positions & facings
    facing_map = {"up": 4, "down": 5, "left": 6, "right": 7}  # channel index for each facing
    for pos in state["wumpus"]:
        fr, fc = pos
        fac = state["wumpus_facing"].get(pos, "down")
        ch[facing_map[fac], fr, fc] = 1.0

    have_gold = 1.0 if state.get("have_gold", False) else 0.0

    # convert to torch tensor (C, H, W)
    t_ch = torch.from_numpy(ch)
    return t_ch, torch.tensor([have_gold], dtype=torch.float32)


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden=256, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def run_episode(env, policy, device):
    """Run one episode with current policy. Return list of (state_vec, action) and total reward."""
    traj = []
    state = env.get_state()
    done = state["done"]
    total_reward = 0
    while not done:
        t_ch, have_gold = state_to_tensor(state, env.h, env.w)  # (C, H, W), (1,)
        inp = torch.cat((t_ch.flatten(), have_gold)).to(device)  # (N,)
        with torch.no_grad():
            logits = policy(inp.unsqueeze(0))  # (1, 4)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
            action = torch.multinomial(probs, num_samples=1).item()
        reward, done = env.step(action)
        total_reward += reward
        traj.append((inp.cpu(), action))
        state = env.get_state()
    return traj, total_reward


def train_cem(
    env,
    n_iterations=200,
    batch_size=64,  # episodes per iteration
    elite_frac=0.2,
    epochs=3,
    lr=1e-3,
    device=torch.device("cpu"),
):
    H, W = env.h, env.w
    C = 8
    input_dim = C * H * W + 1  # flattened channels + have_gold scalar
    policy = PolicyNet(input_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for it in range(n_iterations):
        episodes = []
        rewards = []
        for _ in range(batch_size):
            env.reset()
            traj, total = run_episode(env, policy, device)
            episodes.append(traj)
            rewards.append(total)

        rewards = np.array(rewards)
        # select elites
        k = max(1, int(batch_size * elite_frac))
        elite_idx = rewards.argsort()[-k:]
        elite_trajs = [episodes[i] for i in elite_idx]

        # gather state/action pairs from elites
        states = []
        actions = []
        for traj in elite_trajs:
            for s_vec, a in traj:
                states.append(s_vec.numpy())
                actions.append(a)

        if len(states) == 0:
            print(f"Iter {it}: no elite states collected, skipping update")
            continue

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(device)  # (N, input_dim)
        actions = torch.tensor(actions, dtype=torch.long).to(device)  # (N,)

        # train on elite dataset
        for ep in range(epochs):
            optimizer.zero_grad()
            logits = policy(states)  # (N, 4)
            loss = loss_fn(logits, actions)
            loss.backward()
            optimizer.step()

        # logging
        mean_r = rewards.mean()
        best_r = rewards.max()
        print(f"Iter {it:03d}  meanR {mean_r:.1f}  bestR {best_r:.1f}  elite_size {len(states)}")

    return policy
