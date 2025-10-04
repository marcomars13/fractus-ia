import matplotlib.pyplot as plt

# Séquence "saine"
data_saine = [50,51,49,50,52,48,51,49,50,50,51,49,50,50,49,51,50,49,50,50,51,49,50,50,49,51,50,50,50,49]

# Séquence avec "mutation" (valeur aberrante 120 au milieu)
data_mutation = [50,51,49,50,52,48,51,49,50,50,51,49,50,120,49,51,50,49,50,50,51,49,50,50,49,51,50,50,50,49]

# Scores fractus (issus de tes tests)
scores_saine = [0.961,0.981,0.942,0.962,1.0,0.923,0.981,0.942,0.962,0.962,
                0.981,0.942,0.962,0.962,0.942,0.981,0.962,0.942,0.962,0.962,
                0.981,0.942,0.962,0.962,0.942,0.981,0.962,0.962,0.962,0.942]

scores_mutation = [0.417,0.425,0.408,0.417,0.433,0.400,0.425,0.408,0.417,0.417,
                   0.425,0.408,0.417,1.000,0.408,0.425,0.417,0.408,0.417,0.417,
                   0.425,0.408,0.417,0.417,0.408,0.425,0.417,0.417,0.417,0.408]

# Plot
plt.figure(figsize=(10,5))
plt.plot(scores_saine, label="Séquence saine", color="blue", marker="o")
plt.plot(scores_mutation, label="Séquence avec mutation", color="red", marker="x")
plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)

plt.title("Détection d’anomalie par Fractus")
plt.xlabel("Position dans la séquence")
plt.ylabel("Score Fractus")
plt.legend()
plt.grid(True)

# Sauvegarde dans un fichier
plt.savefig("fractus_demo.png", dpi=150)

# Affiche automatiquement le graphique dans une fenêtre
plt.show()

print("✅ Graphique généré et affiché.")

