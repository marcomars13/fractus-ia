import os, csv

images_dir = "/Users/marco/Projets/fractus-ia/data/mapillary_world/thumbs_clean"
gt_file = "/Users/marco/Projets/fractus-ia/data/mapillary_world/ground_truth_world_matched.csv"

# fichiers réels
imgs = set(os.listdir(images_dir))

# GT depuis CSV
with open(gt_file, newline="") as f:
    reader = csv.DictReader(f)
    gt_names = {row["filename"].strip() for row in reader}

matches = imgs & gt_names
only_in_imgs = imgs - gt_names
only_in_gt = gt_names - imgs

print(f"📊 Images dans dossier : {len(imgs)}")
print(f"📊 Entrées dans GT     : {len(gt_names)}")
print(f"✅ Correspondances     : {len(matches)}")

print("\nExemples présents dans dossier mais pas dans GT:")
print(list(only_in_imgs)[:10])

print("\nExemples présents dans GT mais pas dans dossier:")
print(list(only_in_gt)[:10])


