import json
import statistics
from pathlib import Path

# 📂 Fichiers d'entrée/sortie
INPUT_FILE = Path("/Users/marco/Desktop/bench_direct_plonk_fractus_results.json")
OUTPUT_FILE = Path("/Users/marco/Desktop/bench_direct_plonk_fractus_summary.json")

def main():
    if not INPUT_FILE.exists():
        print(f"❌ Fichier introuvable: {INPUT_FILE}")
        return

    # Charger les résultats bruts
    with open(INPUT_FILE, "r") as f:
        results = json.load(f)

    plonk_confidences = []
    fractus_scores = []

    for item in results:
        # Sécurité: on récupère uniquement si dispo
        plonk_conf = item.get("plonk", {}).get("confidence")
        fractus_score = item.get("fractus", {}).get("score")

        if plonk_conf is not None:
            plonk_confidences.append(plonk_conf)
        if fractus_score is not None:
            fractus_scores.append(fractus_score)

    summary = {
        "n_images": len(results),
        "plonk_confidence": {
            "mean": round(statistics.mean(plonk_confidences), 4) if plonk_confidences else None,
            "median": round(statistics.median(plonk_confidences), 4) if plonk_confidences else None,
            "min": round(min(plonk_confidences), 4) if plonk_confidences else None,
            "max": round(max(plonk_confidences), 4) if plonk_confidences else None,
        },
        "fractus_scores": {
            "mean": round(statistics.mean(fractus_scores), 4) if fractus_scores else None,
            "median": round(statistics.median(fractus_scores), 4) if fractus_scores else None,
            "min": round(min(fractus_scores), 4) if fractus_scores else None,
            "max": round(max(fractus_scores), 4) if fractus_scores else None,
        }
    }

    # Sauvegarder le résumé JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    # Afficher un petit résumé terminal
    print("✅ Analyse terminée")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

