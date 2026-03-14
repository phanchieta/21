import numpy as np

def save_q_table(q_table, filename="blackjack_brain.npy"):
    # Convert dictionary to a numpy object array for saving
    np.save(filename, q_table)
    print(f"Model saved to {filename}")

def load_q_table(filename="blackjack_brain.npy"):
    try:
        # Allow_pickle is needed because our keys are tuples
        return np.load(filename, allow_pickle=True).item()
    except FileNotFoundError:
        print("No saved model found. Starting with a fresh brain.")
        return {}