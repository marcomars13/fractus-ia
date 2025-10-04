import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "backend/fractus_skyline_scores.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    print("📊 Stats scores Skyline")
    print(df["score"].describe())

    # Histogramme
    plt.figure(figsize=(6,4))
    plt.hist(df["score"], bins=10, edgecolor="black")
    plt.title("Distribution des scores Skyline Fractus")
    plt.xlabel("Score")
    plt.ylabel("Nombre d'images")
    plt.tight_layout()
    plt.savefig("backend/fractus_skyline_hist.png")
    plt.close()

    # Scatter plot
    plt.figure(figsize=(6,4))
    plt.scatter(range(len(df)), df["score"], color="blue", label="Scores")
    plt.title("Scores Skyline Fractus (par image)")
    plt.xlabel("Index image")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("backend/fractus_skyline_scatter.png")
    plt.close()

    print("📂 Graphes sauvegardés : backend/fractus_skyline_hist.png et backend/fractus_skyline_scatter.png")

if __name__ == "__main__":
    main()


