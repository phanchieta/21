import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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