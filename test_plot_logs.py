import numpy as np
import matplotlib.pyplot as plt

# === Génération de logs simulés ===
np.random.seed(42)
logs = np.random.normal(loc=50, scale=5, size=200)

# Ajout de quelques anomalies artificielles
logs[50] += 30   # pic fort
logs[123] += 15  # anomalie moyenne
logs[175] -= 25  # chute brutale

# === Fonction Fractus simplifiée ===
def fractus_scores(data, window=7):
    scores = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        local = data[start:end]
        score = abs(data[i] - np.mean(local)) / (np.std(local) + 1e-6)
        scores.append(min(score, 1.0))
    return np.array(scores)

scores = fractus_scores(logs, window=7)

# === Création du dashboard (3 panneaux) ===
fig, (ax0, ax1, ax2) = plt.subplots(
    3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1.5, 1]}
)

# --- Panneau 1 : Logs bruts ---
ax0.plot(logs, color="gray", label="Logs bruts")
ax0.set_title("Série de logs simulés")
ax0.set_ylabel("Valeur brute")
ax0.legend()

# --- Panneau 2 : Scores Fractus ---
ax1.plot(scores, label="Scores Fractus", color="purple")
ax1.set_title("Détection d’anomalies avec Fractus")
ax1.set_xlabel("Position")
ax1.set_ylabel("Score Fractus")
ax1.legend()

# Annotation : top 5 anomalies
top_idx = scores.argsort()[-5:][::-1]
for i in top_idx:
    ax1.plot(i, scores[i], "rx")
    ax1.text(i, scores[i] + 0.03, f"{i}", color="red", ha="center")

# --- Panneau 3 : Explication IA ---
explanation = []
max_score = np.max(scores)
if max_score > 0.9:
    explanation.append(f"⚠ Pic majeur détecté à l’index {np.argmax(scores)} avec score = {max_score:.2f}.")
if np.std(scores) > 0.1:
    explanation.append("⚠ Variabilité notable : fluctuations significatives dans les logs.")
if np.mean(scores) < 0.6:
    explanation.append("✅ Tendance globale rassurante : la majorité des scores restent modérés.")
if not explanation:
    explanation.append("✅ Pas d’anomalies significatives détectées.")

ax2.axis("off")
ax2.set_title("Analyse IA des résultats Fractus", fontsize=12, loc="left")
ax2.text(0, 1, "\n".join(explanation), fontsize=11, va="top", wrap=True)

plt.tight_layout()
plt.savefig("fractus_logs_dashboard_full.png")
plt.show()

print("✅ Dashboard généré : fractus_logs_dashboard_full.png")

