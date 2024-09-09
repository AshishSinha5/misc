import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class NonStationaryBandit:
    def __init__(self, k = 10):
        self.k = k
        self.time = 0
        self.true_values = np.zeros(k)
        self.reset()

    def reset(self):
        self.time = 0
        self.true_values = np.random.normal(0, 1, self.k)

    def pull(self, arm):
        self.time += 1
        self.true_values+= np.random.normal(0, 0.001, self.k)
        return np.random.normal(self.true_values[arm],1)

    
class NonStationaryAgent:
    def __init__(self, k, alpha=0.1, epsilon=0.1):
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.q_values = np.zeros(self.k)
        self.action_counts = np.zeros(self.k)

    def choose_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += self.alpha * (reward - self.q_values[action])
        
def run_experiment(agent, environment, num_steps):
    total_reward = 0
    rewards = []
    for _ in tqdm(range(num_steps)):
        action = agent.choose_action()
        reward = environment.pull(action)
        agent.update(action, reward)
        rewards.append(reward)
        total_reward += reward
    return total_reward, rewards


num_arms = 10
num_steps = 100000
alpha = 0.1
epsilon = 0.1

bandit = NonStationaryBandit(num_arms)
agent = NonStationaryAgent(num_arms, alpha, epsilon)


total_reward, rewards = run_experiment(agent, bandit, num_steps)

plt.plot(np.cumsum(rewards)/(np.arange(num_steps) + 1))
plt.ylabel("Average Reawrd over time")
plt.xlabel("Num Steps")
plt.savefig("plots/non_stationary_bandit.png")

print(f"{total_reward = }")
print(f"{agent.q_values = }")
