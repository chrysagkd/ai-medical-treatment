import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# --- Περιβάλλον HIV Therapy ---
class HIVTherapyEnv(gym.Env):
    def __init__(self):
        super(HIVTherapyEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.max_steps = 50
        self.step_count = 0

    def reset(self):
        self.state = np.array([80.0, 20.0], dtype=np.float32)
        self.step_count = 0
        return self.state

    def step(self, action):
        viral_load, immune_cells = self.state
        if action == 0:
            viral_load += random.uniform(5, 10)
            immune_cells -= random.uniform(1, 3)
        elif action == 1:
            viral_load -= random.uniform(3, 6)
            immune_cells += random.uniform(2, 4)
        else:
            viral_load -= random.uniform(7, 12)
            immune_cells -= random.uniform(2, 5)

        viral_load = np.clip(viral_load, 0, 100)
        immune_cells = np.clip(immune_cells, 0, 100)
        self.state = np.array([viral_load, immune_cells], dtype=np.float32)

        reward = (100 - viral_load) + immune_cells - 50
        self.step_count += 1
        done = self.step_count >= self.max_steps or immune_cells <= 0 or viral_load >= 100
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Step {self.step_count}: Viral Load={self.state[0]:.1f}, Immune Cells={self.state[1]:.1f}")

# --- Q-learning Agent ---
class QAgent:
    def __init__(self, bins=(10, 10), n_actions=3, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.bins = bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros(self.bins + (n_actions,))

    def discretize(self, state):
        ratios = state / 100.0
        indices = (ratios * np.array(self.bins)).astype(int)
        return tuple(np.clip(indices, 0, np.array(self.bins) - 1))

    def select_action(self, state):
        d_state = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[d_state])

    def learn(self, state, action, reward, next_state, done):
        d_state = self.discretize(state)
        d_next = self.discretize(next_state)
        target = reward + (0 if done else self.gamma * np.max(self.q_table[d_next]))
        self.q_table[d_state][action] += self.alpha * (target - self.q_table[d_state][action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Εκπαίδευση με καταγραφή ---
def train_agent(episodes=300):
    env = HIVTherapyEnv()
    agent = QAgent()
    rewards = []
    epsilons = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}: Total Reward={total_reward:.2f} | Epsilon={agent.epsilon:.3f}")

    return agent, env, rewards, epsilons

# --- Οπτικοποίηση της εκπαίδευσης ---
def plot_training(rewards, epsilons):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color='tab:blue')
    ax1.plot(rewards, label='Total Reward', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon', color='tab:red')
    ax2.plot(epsilons, label='Epsilon', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.title('Training Progress')
    fig.tight_layout()
    plt.show()

# --- Demo Agent με animation και treatment info ---
def demo(agent, env):
    state = env.reset()
    viral_list = [state[0]]
    immune_list = [state[1]]
    steps = [0]
    actions_taken = []

    action_names = {0: "No Treatment", 1: "Low Dose", 2: "High Dose"}

    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], label="Viral Load", color='yellow')
    line2, = ax.plot([], [], label="Immune Cells", color='black')
    ax.set_xlim(0, env.max_steps)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Levels")
    ax.legend()
    plt.title("HIV Therapy Demo")

    text_action = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def update(frame):
        nonlocal state
        action = agent.select_action(state)
        next_state, _, done, _ = env.step(action)
        viral_list.append(next_state[0])
        immune_list.append(next_state[1])
        steps.append(frame + 1)
        actions_taken.append(action)

        line1.set_data(steps, viral_list)
        line2.set_data(steps, immune_list)

        text_action.set_text(f"Step {frame+1}: Treatment = {action_names[action]}")

        state = next_state
        return line1, line2, text_action

    ani = animation.FuncAnimation(fig, update, frames=env.max_steps - 1, interval=300, blit=True, repeat=False)
    plt.show()

# --- Εκκίνηση ---
if __name__ == "__main__":
    agent, env, rewards, epsilons = train_agent(episodes=300)
    plot_training(rewards, epsilons)
    demo(agent, env)
