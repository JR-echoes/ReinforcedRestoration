import numpy as np
import random
import PowerNetwork
import pypsa

##LETS START WITH A SIMPLE RL MODEL WITH THREE STATES, COLLAPSED, PARTIALLY RESTORED AND FULLY RESTORED

class PowerGridEnvironment:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states #number of states
        self.num_actions = num_actions # number of actions to take
        self.state = 0  # Start with a collapsed grid

    def reset(self):
        self.state = 0  # Reset to collapsed state
        return self.state

    def step(self, action):
        # Let's define how actions affect the state and the acquired reward
        if self.state == 0:  # Collapsed state
            if action == 0:  # option exmple, lets say charge a line( increases the no of customer fed or no of buses charged or volatage/frequency stability)
                ## Here, we need to perform mini power flow analysis before deciding reward or fine
                reward = 10
                self.state = 1  # Transition to partially restored next state
            else:
                reward = -1  # negative reward for wrong action
        elif self.state == 1: 
            if action == 1:  # Action to complete restoration
                 ## Here, we need to perform mini power flow analysis before deciding reward or fine
                reward = 100
                self.state = 2  # Transition to fully restored state
            else:
                reward = -1
        else:
            reward = 0  # No reward given in a fully restored power grid state

        done = (self.state == 2)
        return self.state, reward, done

    def get_state_space(self):
        return self.num_states

    def get_action_space(self):
        return self.num_actions


class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.num_actions))  # Explore the options
        else:
            return np.argmax(self.q_table[state])  # Exploit the options

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

def train_agent(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            print(f"Episode: {episode}, Action: {action}, Next State: {state}, Reward: {reward}\n")
            agent.update_q_table(state, action, reward, next_state)
            state = next_state

if __name__ == "__main__":
    print(f"REINFORCED LEARNING TO RESTORE A COLLAPSED GRID. TRIAL: MINI TRAINING AND MINI POWER FLOW ANALYSIS")
    num_states = 3  # 0=collapsed, 1=partially restored, 2=fully restored
    num_actions = 2  # Example: charge/discharge component, complete restoration
    env = PowerGridEnvironment(num_states, num_actions)
    agent = QLearningAgent(num_states, num_actions)

    train_agent(env, agent, num_episodes=100) ##Let's train the model agent for 100 episodes

    # Let's test the trained model now
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        print(f"State: {state}, Action: {action}, Reward: {reward}")
