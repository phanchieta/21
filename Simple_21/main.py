from load_save import load_q_table
from train import train
from visualize import plot_strategy
import gymnasium as gym
import numpy as np

def get_q_values(state):
    if state not in q_table:
        q_table[state] = [0.0, 0.0]  # Start with no knowledge
    return q_table[state]


print("Trying to load model..")
q_table=load_q_table(filename="checkpoints\\blackjack_brain.npy")
if not q_table:
    train()
    q_table=load_q_table(filename="checkpoints\\blackjack_brain.npy")
else:
    print("Model loaded!")

# Evaluation Loop
wins, losses, draws = 0, 0, 0
test_episodes = 10000

env = gym.make('Blackjack-v1', natural=False, sab=False)

for _ in range(test_episodes):
    state, info = env.reset()
    done = False
    
    while not done:
        # No more epsilon! Always take the best move (Argmax)
        action = np.argmax(get_q_values(state))
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        draws += 1

print(f"--- Results after {test_episodes} games ---")
print(f"Win Rate:  {(wins/test_episodes)*100:.2f}%")
print(f"Loss Rate: {(losses/test_episodes)*100:.2f}%")
print(f"Draw Rate: {(draws/test_episodes)*100:.2f}%")

plot_strategy(q_table)