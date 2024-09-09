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
    
    def get_optimal_actions(self):
        return np.argmax(self.true_values)

    
class NonStationaryAgent:
    def __init__(self, k, alpha=0.1, epsilon=0.1, init_val = 0):
        self.k = k
        self.alpha = alpha
        self.epsilon = epsilon
        self.init_val = init_val
        self.reset()

    def reset(self):
        self.q_values = np.ones(self.k)*self.init_val
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
    optimal_actions = []
    for _ in tqdm(range(num_steps)):
        optimal_arms = environment.get_optimal_actions()
        action = agent.choose_action()
        reward = environment.pull(action)
        agent.update(action, reward)
        rewards.append(reward)
        total_reward += reward
        optimal_actions.append(1 if optimal_arms == action else 0)
    return total_reward, rewards, optimal_actions


num_arms = 10
num_steps = 10000
alpha = 0.1
epsilon = 0.1

bandit = NonStationaryBandit(num_arms)
agent = NonStationaryAgent(num_arms, alpha, epsilon)


total_reward, rewards, optimal_actions_eps = run_experiment(agent, bandit, num_steps)

plt.plot(np.cumsum(rewards)/(np.arange(num_steps) + 1), label="eps_greedy")

print(f"{total_reward = }")
print(f"{agent.q_values = }")

epsilon = 0
init_val = 5

bandit = NonStationaryBandit(num_arms)
agent = NonStationaryAgent(num_arms, alpha, epsilon, init_val)

total_reward, rewards, optimal_actions_greedy = run_experiment(agent, bandit, num_steps)

plt.plot(np.cumsum(rewards)/(np.arange(num_steps) + 1), label="greedy")
plt.ylabel("Average Reawrd over time")
plt.xlabel("Num Steps")
plt.legend()
plt.savefig("plots/non_stationary_bandit.png")
plt.show()
print(f"{total_reward = }")
print(f"{agent.q_values = }")

plt.plot(np.cumsum(optimal_actions_eps)/(np.arange(num_steps) + 1)*100, label="eps with Q1 = 0")
plt.plot(np.cumsum(optimal_actions_greedy)/(np.arange(num_steps) + 1)*100, label="greedy with Q1 = 5")
plt.ylabel("Optimal Actions over time")
plt.xlabel("Num Steps")
plt.legend()
plt.savefig("plots/optimal_actions.png")
