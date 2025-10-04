import os
import pandas as pd

# --- Config ---
IMG_DIR = "test_images"  # 📂 dossier avec tes 10–20 photos de test
OUT_FILE = "data/ground_truth_test.csv"

# ⚠️ À toi de mettre les bonnes coordonnées ici
# Exemple provisoire (Paris, Marseille, Lyon, Nice)
ground_truth = {
    "photo1": (48.8566, 2.3522),   # Paris
    "photo2": (43.2965, 5.3698),   # Marseille
    "photo3": (45.7640, 4.8357),   # Lyon
    "photo4": (43.7102, 7.2620),   # Nice
}

# Vérifier dossier
if not os.path.exists(IMG_DIR):
    print(f"⚠️ Dossier {IMG_DIR} introuvable")
    exit(1)

rows = []
for fname in os.listdir(IMG_DIR):
    if fname.lower().endswith(".jpg"):
        stem = os.path.splitext(fname)[0]
        if stem in ground_truth:
            lat, lon = ground_truth[stem]
            rows.append({"filename": stem, "lat": lat, "lon": lon})
        else:
            print(f"⚠️ Pas de GT défini pour {fname}")

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
df.to_csv(OUT_FILE, index=False)

print(f"📂 CSV de test sauvegardé : {OUT_FILE}")
print(df.head())

