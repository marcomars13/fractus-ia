"""
multi_infer.py — Empilement temporel / multi-photos pour Fractus+Plonk

Idée:
- On passe N photos du "même" lieu (angles/moments différents).
- Pour chaque photo: on génère des candidats (lat, lon, score_fractus, score_final).
- On "vote" en agrégeant les candidats proches (clustering par rayon en km).
- On renvoie un consensus robuste trié par score agrégé.

Dépendances internes :
- constraints.predict_hybrid (ne casse rien si flags OFF)
- pipeline.run_plonk_fractus est déjà appelé *via* constraints

Sortie :
- Liste de dicts: {lat, lon, votes, score_mean, score_median, score_final}
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import math

# ---- Imports internes sûrs
try:
    from constraints import predict_hybrid, Flags
except Exception:
    predict_hybrid = None
    Flags = None


# ------------------------------
# Utils géo
# ------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ------------------------------
# Clustering par rayon
# ------------------------------
@dataclass
class Cluster:
    lat_sum: float = 0.0
    lon_sum: float = 0.0
    votes: int = 0
    scores: List[float] = None

    def add(self, lat: float, lon: float, score: float):
        if self.scores is None:
            self.scores = []
        self.lat_sum += lat
        self.lon_sum += lon
        self.votes += 1
        self.scores.append(float(score))

    def centroid(self) -> Tuple[float, float]:
        if self.votes == 0:
            return (0.0, 0.0)
        return (self.lat_sum / self.votes, self.lon_sum / self.votes)

    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def median_score(self) -> float:
        if not self.scores:
            return 0.0
        s = sorted(self.scores)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2.0


def cluster_candidates(cands: List[Tuple[float, float, float, float]], radius_km: float = 50.0):
    """
    cands: liste de tuples (lat, lon, score_fractus, score_final)
    Retourne: liste de Cluster agrégés
    """
    clusters: List[Cluster] = []
    for lat, lon, s_frac, s_final in cands:
        placed = False
        for cl in clusters:
            cl_lat, cl_lon = cl.centroid()
            if haversine_km(lat, lon, cl_lat, cl_lon) <= radius_km:
                cl.add(lat, lon, s_final)
                placed = True
                break
        if not placed:
            cl = Cluster()
            cl.add(lat, lon, s_final)
            clusters.append(cl)
    return clusters


# ------------------------------
# Multi-infer principal
# ------------------------------
def run_multi_infer(
    image_paths: List[str],
    top_k: int = 5,
    radius_km: float = 50.0,
    flags: "Flags" = None,
) -> List[Dict[str, Any]]:
    """
    Empilement multi-photos.
    - image_paths : chemins vers les images (peuvent être différentes ou redondantes)
    - top_k       : on prend top_k candidats par image (via constraints.predict_hybrid)
    - radius_km   : rayon de clustering pour agréger les votes
    - flags       : constraints.Flags (tous OFF -> status quo)

    Retour: liste de consensus triée par score_final agrégé (mean + bonus votes)
    """
    if predict_hybrid is None or Flags is None:
        # fallback safe
        return []

    f = flags if flags is not None else Flags()  # garde tes flags système
    collected: List[Tuple[float, float, float, float]] = []

    # 1) récupérer candidats pour chaque image
    for path in image_paths:
        try:
            results = predict_hybrid(path, f)  # [(lat,lon,score_fractus,score_final), ...]
        except Exception:
            results = []
        # Limiter à top_k si la liste est longue
        results = list(results)[:top_k]
        collected.extend(results)

    if not collected:
        return []

    # 2) clusteriser par proximité géographique
    clusters = cluster_candidates(collected, radius_km=radius_km)

    # 3) scorer les clusters (consensus)
    out: List[Dict[str, Any]] = []
    for cl in clusters:
        lat_c, lon_c = cl.centroid()
        mean_s = cl.mean_score()
        med_s = cl.median_score()
        # score final : moyenne + petit bonus votes (logarithmique pour éviter de tout écraser)
        consensus = mean_s + 0.05 * math.log(max(cl.votes, 1) + 1.0)
        out.append(
            {
                "lat": round(lat_c, 6),
                "lon": round(lon_c, 6),
                "votes": cl.votes,
                "score_mean": round(mean_s, 6),
                "score_median": round(med_s, 6),
                "score_final": round(consensus, 6),
            }
        )

    # 4) trier du meilleur au moins bon
    out.sort(key=lambda d: d["score_final"], reverse=True)
    return out


# ------------------------------
# Entrée type API (facultatif)
# ------------------------------
def api_multi_infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Payload attendu :
    {
      "images": ["a.jpg", "b.jpg", ...],
      "top_k": 5,
      "radius_km": 50,
      "flags": {"use_solar":0,"use_dem":0,"use_calib":1}
    }
    """
    imgs = payload.get("images", [])
    top_k = int(payload.get("top_k", 5))
    radius_km = float(payload.get("radius_km", 50.0))
    flags_dict = payload.get("flags", {})

    # construire Flags sans casser la signature existante
    f = Flags(
        use_solar=bool(flags_dict.get("use_solar", False)),
        use_dem=bool(flags_dict.get("use_dem", False)),
        use_calib=bool(flags_dict.get("use_calib", False)),
    )

    results = run_multi_infer(imgs, top_k=top_k, radius_km=radius_km, flags=f)
    return {"ok": True, "count": len(results), "results": results}

