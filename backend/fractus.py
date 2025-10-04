import numpy as np

def compute_fractus_scores(image, window=32, pattern="default"):
    """
    Calcule les scores Fractus pour une image donnée.
    
    Args:
        image (np.ndarray): Image en entrée (BGR ou RGB).
        window (int): Taille de fenêtre pour l’analyse locale.
        pattern (str): Motif fractal à utiliser ("default" par défaut).
    
    Returns:
        np.ndarray: Tableau de scores numériques.
    """
    try:
        # Vérification image
        if image is None:
            raise ValueError("Image vide ou non lisible")

        # Conversion en niveaux de gris si nécessaire
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float32)

        # Exemple simplifié : découpage en fenêtres et calcul de variance locale
        h, w = gray.shape
        scores = []

        for y in range(0, h - window, window):
            for x in range(0, w - window, window):
                patch = gray[y:y+window, x:x+window]
                if patch.size == 0:
                    continue
                scores.append(np.var(patch))

        scores = np.array(scores, dtype=np.float32)

        # ✅ Correction : éviter le bug "ValueError: truth value ambiguous"
        if scores is None or scores.size == 0:
            return None

        return scores

    except Exception as e:
        print(f"⚠️ [compute_fractus_scores] Erreur: {e}")
        return None

