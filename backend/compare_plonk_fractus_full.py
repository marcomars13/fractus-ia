import os, csv, json, warnings
from tqdm import tqdm
from plonk import PlonkPipeline
from fractus_ultimate import run_fractus_ultimate
from PIL import Image

# 🔇 Désactiver les warnings xFormers
warnings.filterwarnings("ignore", message="xFormers is not available")

def main():
    print("⚡ Fractus Ultimate (KDTree + vote) prêt")
    print("🚀 Comparaison Plonk officiel (HQ) vs Fractus Ultimate")

    img_dir = "test_images"
    gt_file = "data/ground_truth_subset.csv"

    images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]
    print(f"📁 Dossier images retenu : {img_dir}\n   → {len(images)} .jpg détectés")

    # Charger Plonk officiel
    print("🚀 Chargement PlonkPipeline (YFCC) device=cpu steps=120 cfg=0.0")
    pipe = PlonkPipeline("nicolas-dufour/PLONK_YFCC", device="cpu")

    rows = []
    for img in tqdm(images, desc="Comparaison"):
        try:
            # Charger et redimensionner l’image (mais rester en PIL)
            pil_img = Image.open(img).convert("RGB")
            pil_img = pil_img.resize((224,224))  # 🔧 Resize uniquement

            # ✅ Prédiction Plonk
            plonk_pred = pipe([pil_img])
            lat, lon = plonk_pred[0].tolist()
            plonk = {"lat": lat, "lon": lon}
        except Exception as e:
            print(f"⚠️ Erreur Plonk sur {img}: {e}")
            plonk = {"lat": None, "lon": None}

        try:
            fractus = run_fractus_ultimate(img)
        except Exception as e:
            print(f"⚠️ Erreur Fractus sur {img}: {e}")
            fractus = {"lat": None, "lon": None}

        rows.append({
            "image": os.path.basename(img),
            "plonk_lat": plonk["lat"],
            "plonk_lon": plonk["lon"],
            "fractus_lat": fractus["lat"],
            "fractus_lon": fractus["lon"],
        })

    print(f"🔎 Nombre de résultats collectés: {len(rows)}")

    if not rows:
        print("⚠️ Aucun résultat généré → vérifie tes images ou modèles")
        return

    # Sauvegarde CSV
    out_csv = "backend/plonk_fractus_full_results.csv"
    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Sauvegarde JSON résumé
    out_json = "backend/plonk_fractus_full_summary.json"
    with open(out_json, "w") as fjson:
        json.dump(rows, fjson, indent=2)

    print(f"\n📂 Résultats sauvegardés dans {out_csv}")
    print(f"📂 Résumé sauvegardé dans {out_json}")

if __name__ == "__main__":
    main()

