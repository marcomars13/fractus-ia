"""
auto_selector.py — Sélection automatique des flags (version corrigée)
Règles :
- Ville banale → Fractus OFF (laisser Plonk seul)
- Ville structurée → Fractus ON hybride
- Montagne/nature → Fractus pur fallback
"""

from constraints import Flags, predict_hybrid, VECTOR_DB
from skyline_enhancer import skyline_match
from spectral_analyzer import analyze_spectrum
from vector_match import dummy_encode


def classify_scene(image_path: str) -> str:
    """
    Détermine le type de scène : 'banale', 'structurée', 'montagne'.
    Logique corrigée :
    - Spectral = forêt/dune → priorité absolue 'montagne'
    - Skyline = montagne → 'montagne'
    - Spectral = urbain/périodique → 'structurée'
    - Sinon → 'banale'
    """
    # Spectral en priorité
    try:
        spec = analyze_spectrum(image_path)
        cat = spec.get("category")
        if cat == "forêt/dune":
            return "montagne"
        elif cat == "urbain/périodique":
            return "structurée"
    except Exception:
        pass

    # Skyline si pas déjà tranché
    try:
        res = skyline_match(image_path)
        skyline_label = res.get("label_best")
        if skyline_label == "montagne":
            return "montagne"
    except Exception:
        pass

    return "banale"


def auto_select_flags(image_path: str, verbose: bool = True) -> Flags:
    """Active dynamiquement les flags selon le type de scène détecté."""
    scene = classify_scene(image_path)
    if verbose:
        print(f"[AUTO] Scène détectée : {scene}")

    # Flags de base
    f = Flags(use_solar=1, use_dem=1, use_calib=1)

    # Ville banale → Fractus OFF
    if scene == "banale":
        if verbose:
            print("[AUTO] Ville banale → Fractus OFF (Plonk seul)")
        return f

    # Ville structurée → Fractus hybride
    if scene == "structurée":
        f.use_skyline = True
        f.use_spectral = True
        try:
            if VECTOR_DB is not None and dummy_encode is not None:
                qvec = dummy_encode(image_path)
                matches = VECTOR_DB.search(qvec, top_k=1)
                if matches and matches[0]["distance"] < 1.5:
                    f.use_vector = True
        except Exception:
            pass
        if verbose:
            print("[AUTO] Ville structurée → Fractus hybride activé")
        return f

    # Montagne/nature → Fractus pur fallback
    if scene == "montagne":
        f.use_fallback = True
        if verbose:
            print("[AUTO] Montagne/nature → Fractus pur fallback")
        return f

    return f


def auto_predict(image_path: str, verbose: bool = True):
    """Prédiction automatique avec règles contextuelles."""
    flags = auto_select_flags(image_path, verbose=verbose)
    preds = predict_hybrid(image_path, flags)
    return flags, preds

