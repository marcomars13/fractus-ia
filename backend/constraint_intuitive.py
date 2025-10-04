import numpy as np

def apply_constraints(prediction, neighbors, skyline_hint=None, cluster_radius_km=200):
    """
    Applique une contrainte intuitive sur une prédiction géographique.

    Args:
        prediction (dict): {"lat": float, "lon": float, "score": float}
        neighbors (list): liste de (lat, lon, score) pour les top_k voisins
        skyline_hint (str): optionnel, ex. "flat" ou "mountain"
        cluster_radius_km (float): rayon max pour considérer un cluster cohérent

    Returns:
        dict: prédiction possiblement corrigée {"lat":, "lon":, "score":, "info":}
    """

    # --- Contrainte 1: cluster géographique ---
    # On cherche le cluster le plus dense parmi les voisins
    cluster = []
    for i, (lat, lon, score) in enumerate(neighbors):
        count = 0
        for j, (lat2, lon2, _) in enumerate(neighbors):
            d = haversine(lat, lon, lat2, lon2)
            if d < cluster_radius_km:
                count += 1
        cluster.append((lat, lon, score, count))

    # On choisit le point du cluster le plus dense
    cluster_sorted = sorted(cluster, key=lambda x: (x[3], -x[2]), reverse=True)
    best_cluster = cluster_sorted[0]

    # --- Contrainte 2: skyline intuitive ---
    info = "cluster"
    if skyline_hint is not None:
        if skyline_hint == "flat":
            info += " + skyline(flat)"
            # Si horizon plat, on pénalise les zones montagneuses (lat > 45N ou alt supposée)
            if abs(best_cluster[0]) > 45:
                # on garde la prédiction d'origine
                return {**prediction, "info": info + " (préservation prédiction)"}

    # Sinon, on retourne le cluster le plus dense
    return {
        "lat": best_cluster[0],
        "lon": best_cluster[1],
        "score": best_cluster[2],
        "info": info
    }


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

