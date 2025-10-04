import matplotlib.pyplot as plt
import requests
import json

# Séquence avec "mutation"
data_mutation = [50,51,49,50,52,48,51,49,50,50,51,49,50,120,49,51,50,49,50,50,
                 51,49,50,50,49,51,50,50,50,49]

# Paramètres API
payload = {"data": data_mutation, "window": 7, "pattern": ""}

# Appel API explication
url_explain = "http://127.0.0.1:8000/anomaly/explain"
try:
    r = requests.post(url_explain, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    r.raise_for_status()
    result = r.json()
    explanation = result.get("explanation", "Pas d'explication renvoyée.")
except Exception as e:
    explanation = f"❌ Erreur API: {e}"

# Appel API scores
url_score = "http://127.0.0.1:8000/anomaly/score"
resp_score = requests.post(url_score, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
scores = resp_score.json().get("scores", [])

# Création figure avec 2 sous-parties (graph + texte)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), gridspec_kw={'height_ratios':[2,1]})

# --- Partie 1 : le graphe ---
ax1.plot(scores, label="Séquence avec mutation", color="red", marker="x")
ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
ax1.set_title("Détection d’anomalie par Fractus", fontsize=14)
ax1.set_xlabel("Position dans la séquence")
ax1.set_ylabel("Score Fractus")
ax1.legend()
ax1.grid(True)

# --- Partie 2 : bloc texte explicatif ---
ax2.axis("off")  # pas d'axes
ax2.text(0, 1, explanation, fontsize=10, wrap=True, va="top")

# Sauvegarde + affichage
plt.tight_layout()
plt.savefig("fractus_explain_gui.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Graphique + explication générés (fractus_explain_gui.png)")

