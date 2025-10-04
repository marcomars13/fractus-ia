"""
pipeline.py — Pont PLONK + Fractus avec encodage image.
"""

from typing import List, Dict, Any

# ---- Imports sûrs ----
try:
    from plonk_infer import plonk_predict
    from fractus_utils import fractus_transform
    from PIL import Image
except Exception:
    plonk_predict = None
    fractus_transform = None
    Image = None


def _encode_image_as_sequence(image_path: str, lat: float, lon: float) -> str:
    """
    Encodage image → séquence pour Fractus :
    - ouvre l’image
    - calcule l’histogramme RGB
    - transforme en string
    - ajoute les coords comme sel
    """
    if Image is None:
        return f"{lat:.6f}|{lon:.6f}"

    try:
        img = Image.open(image_path).convert("RGB")
        hist = img.histogram()  # 768 valeurs (256 par canal)
        # on réduit pour ne pas exploser la taille
        reduced = [sum(hist[i:i+32]) for i in range(0, len(hist), 32)]
        seq = "".join(chr((h % 90) + 33) for h in reduced)  # chars imprimables
        seq = f"{lat:.3f}|{lon:.3f}|" + seq
        return seq
    except Exception:
        return f"{lat:.6f}|{lon:.6f}"


def _adapter_plonk_fractus(image, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Utilise PLONK pour obtenir K candidats, puis calcule un score Fractus
    basé sur histogramme image + coords.
    """
    if plonk_predict is None or fractus_transform is None:
        return [{"lat": 0.0, "lon": 0.0, "score_fractus": 0.5}][:top_k]

    preds = plonk_predict(image, top_k=top_k)

    out: List[Dict[str, Any]] = []
    for p in preds:
        lat = p.get("lat")
        lon = p.get("lon")
        if lat is None or lon is None:
            continue

        seq = _encode_image_as_sequence(image, float(lat), float(lon))
        scores = fractus_transform(seq, window=8) or []
        score = float(sum(scores) / len(scores)) if scores else 0.0

        out.append({
            "lat": float(lat),
            "lon": float(lon),
            "score_fractus": score
        })

    return out


def run_plonk_fractus(image, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Fonction publique utilisée par constraints.py
    """
    results = _adapter_plonk_fractus(image, top_k=top_k)

    clean: List[Dict[str, Any]] = []
    for r in results:
        lat = r.get("lat")
        lon = r.get("lon")
        score = r.get("score_fractus", 1.0)
        if lat is None or lon is None:
            continue
        clean.append({
            "lat": float(lat),
            "lon": float(lon),
            "score_fractus": float(score),
        })
    return clean

