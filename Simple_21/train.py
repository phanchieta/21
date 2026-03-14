import gymnasium as gym
import numpy as np
import random
from load_save import save_q_table


def get_q_values(q_table, state):
    if state not in q_table:
        q_table[state] = [0.0, 0.0]  # Start with no knowledge
    return q_table[state]

# 3. Training
def train():
    # Setup Environment
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    
    # Hyperparameters
    learning_rate = 0.05
    discount_factor = 0.95
    epsilon = 0.2  # 20% chance to explore (try random moves)
    episodes = 500000
    
    # Initialize Q-table: {(state): [q_value_for_stand, q_value_for_hit]}
    q_table = {}
    print("Training...")
    for i in range(episodes):
        state, info = env.reset()
        done = False
    
        while not done:
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = env.action_space.sample() 
            else:
                action = np.argmax(get_q_values(q_table, state)) 
    
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # q learning update
            old_q = get_q_values(q_table, state)[action]
            next_max = np.max(get_q_values(q_table, next_state))
            
            # Bellman Equation update
            new_q = old_q + learning_rate * (reward + (discount_factor * next_max) - old_q)
            q_table[state][action] = new_q
            
            state = next_state
            done = terminated or truncated
    
        # Gradually reduce epsilon (decay) to make it more stable over time
        if i % 10000 == 0:
            epsilon = max(epsilon * 0.99, 0.01)
    
    print(f"Training finished over {episodes} hands.")
    save_q_table(q_table, filename="checkpoints\\blackjack_brain.npy")
    print("Save complete.")


