# helper.py
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

def plot(scores, mean_scores):
    # Save plot every 10 games
    if len(scores) % 10 != 0:
        return

    plt.figure(figsize=(8, 5))
    plt.title(f"Training Progress (Game {len(scores)})")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"plots/training_game_{len(scores)}.png")
    plt.close()
