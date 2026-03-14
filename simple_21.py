import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

def plot_strategy(q_table, usable_ace=False):
    # Filter the Q-table for the specific Ace state
    strategy = np.zeros((10, 10)) # Rows: Player sum (12-21), Cols: Dealer card (1-10)
    
    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            state = (player_sum, dealer_card, usable_ace)
            if state in q_table:
                # 0 = Stand, 1 = Hit. We take the index of the max Q-value.
                strategy[player_sum-12, dealer_card-1] = np.argmax(q_table[state])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(strategy, annot=True, xticklabels=range(1, 11), yticklabels=range(12, 22), 
                cmap="RdYlGn_r", cbar=False)
    plt.title(f"Blackjack AI Strategy (Usable Ace: {usable_ace})")
    plt.xlabel("Dealer Showing Card")
    plt.ylabel("Player Sum")
    plt.show()

# 1. Setup Environment
env = gym.make('Blackjack-v1', natural=False, sab=False)

# 2. Hyperparameters
learning_rate = 0.05
discount_factor = 0.95
epsilon = 0.2  # 20% chance to explore (try random moves)
episodes = 500000

# Initialize Q-table: {(state): [q_value_for_stand, q_value_for_hit]}
q_table = {}

def get_q_values(state):
    if state not in q_table:
        q_table[state] = [0.0, 0.0]  # Start with no knowledge
    return q_table[state]

# 3. Training Loop
for i in range(episodes):
    state, info = env.reset()
    done = False

    while not done:
        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = env.action_space.sample() 
        else:
            action = np.argmax(get_q_values(state)) 

        next_state, reward, terminated, truncated, info = env.step(action)
        
        # q learning update
        old_q = get_q_values(state)[action]
        next_max = np.max(get_q_values(next_state))
        
        # Bellman Equation update
        new_q = old_q + learning_rate * (reward + (discount_factor * next_max) - old_q)
        q_table[state][action] = new_q
        
        state = next_state
        done = terminated or truncated

    # Gradually reduce epsilon (decay) to make it more stable over time
    if i % 10000 == 0:
        epsilon = max(epsilon * 0.99, 0.01)

print(f"Training finished over {episodes} hands.")

plot_strategy(q_table, usable_ace=False)

#print(q_table[(20, 6, False)])
# The value for index 0 (Stand) is MUCH higher than index 1 (Hit).