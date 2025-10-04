import json
import os

DEFAULT_PARAMS = {
    "window": 32,
    "alpha_mean": 0.001,
    "alpha_std": 0.0005,
    "beta_knn": 0.5,
    "beta_plonk": 0.5,
    "pattern": "fractus-default"
}

def load_fractus_profile(path="mapillary_out/fractus_profile.json"):
    """Charge le profil fractus (geo, visual, params).
       Si params n’existe pas, utilise DEFAULT_PARAMS.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Fichier profil introuvable: {path}")

    with open(path, "r") as f:
        profile = json.load(f)

    geo = profile.get("geo", {})
    visual = profile.get("visual", {})
    params = profile.get("params", DEFAULT_PARAMS.copy())

    # Complète avec les valeurs par défaut manquantes
    for k, v in DEFAULT_PARAMS.items():
        params.setdefault(k, v)

    return {
        "geo": geo,
        "visual": visual,
        "params": params
    }


if __name__ == "__main__":
    profile = load_fractus_profile()
    print("✅ Profil chargé")
    print("🌍 Geo:", profile["geo"])
    print("👁️ Visual vector dim:", len(profile["visual"].get("mean_vector", [])))
    print("⚙️ Params:", profile["params"])

