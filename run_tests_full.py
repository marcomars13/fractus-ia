def run_plonk_fractus(image_path, crop_top_ratio=None):
    """
    Pipeline complet Fractus Full :
    - Extraction features fractales (multi-cœurs)
    - Skyline / horizon
    - Mémoire active
    - Contraintes physiques / géographiques
    """

    from backend import fractus_parallel, constraints
    import skyline_enhancer
    import memory

    # Étape 1 : features fractales en parallèle
    features = fractus_parallel.run_fractus_parallel(image_path, n_cores=8)

    # Étape 2 : Skyline / horizon
    features = skyline_enhancer.enhance_skyline(features)

    # Étape 3 : Mémoire active
    features = memory.vector_match(features)

    # Étape 4 : Contraintes physiques/géo
    lat, lon = constraints.apply_constraints(features)

    return lat, lon

