import os, csv, math, random, warnings
from pathlib import Path
from statistics import mean, median
from typing import Callable, Dict, List, Tuple

from PIL import Image
from plonk import PlonkPipeline
from tqdm import tqdm  # ✅ Progress bar

warnings.filterwarnings("ignore")

# -------- utils --------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    import math as m
    p1, p2 = m.radians(lat1), m.radians(lat2)
    dphi = m.radians(lat2 - lat1)
    dlmb = m.radians(lon2 - lon1)
    a = m.sin(dphi/2)**2 + m.cos(p1)*m.cos(p2)*m.sin(dlmb/2)**2
    return 2*R*m.atan2(m.sqrt(a), m.sqrt(1-a))

def load_gt_map(gt_csv: Path) -> Dict[str, Tuple[float,float]]:
    gt = {}
    with open(gt_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("filename") or row.get("image") or ""
            if not name: 
                continue
            stem = Path(name).stem
            gt[stem] = (float(row["lat"]), float(row["lon"]))
    return gt

def list_images(img_dir: Path) -> List[Path]:
    return [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

# -------- prepro variants (PIL in -> PIL out) --------
def identity(img: Image.Image) -> Image.Image:
    return img

def resize_to(sz: int) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        return img.resize((sz, sz), Image.BICUBIC)
    return _fn

def resize_short_side_then_center_crop(side: int) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < h:
            new_w = side
            new_h = int(h * (side / w))
        else:
            new_h = side
            new_w = int(w * (side / h))
        img2 = img.resize((new_w, new_h), Image.BICUBIC)
        # center crop to square
        w2, h2 = img2.size
        left = (w2 - side)//2
        top  = (h2 - side)//2
        return img2.crop((left, top, left+side, top+side))
    return _fn

PREPROS = {
    "pil_identity": identity,
    "resize_224": resize_to(224),
    "resize_336": resize_to(336),
    "resize_384": resize_to(384),
    "resize_518": resize_to(518),
    "shortside_384_center": resize_short_side_then_center_crop(384),
    "shortside_518_center": resize_short_side_then_center_crop(518),
}

def run_plonk(pipe: PlonkPipeline, pil_img: Image.Image):
    out = pipe([pil_img])
    lat, lon = out[0].tolist() if hasattr(out, "tolist") else out[0]
    return float(lat), float(lon)

def main():
    img_dir = Path("test_images")
    gt_csv  = Path("data/ground_truth_subset.csv")
    assert img_dir.exists(), f"Images introuvables: {img_dir}"
    assert gt_csv.exists(),  f"GT introuvable: {gt_csv}"

    gt_map = load_gt_map(gt_csv)

    # ✅ Limite à 5 images max
    imgs = [p for p in list_images(img_dir) if p.stem in gt_map]
    if not imgs:
        print("❌ Aucune image de test ne correspond au GT (stem mismatch).")
        return
    random.seed(0)
    imgs = imgs[:5]
    print(f"🧪 Images testées: {len(imgs)} (rapide)")

    model_name = os.environ.get("PLONK_MODEL", "nicolas-dufour/PLONK_YFCC")
    print(f"🧠 Modèle Plonk: {model_name}")
    pipe = PlonkPipeline(model_name, device="cpu")

    summary = []
    per_variant_errors: Dict[str, List[float]] = {}

    for variant, fn in PREPROS.items():
        errs = []
        print(f"\n🔎 Test variante {variant}...")
        for p in tqdm(imgs, desc=f"{variant}"):
            try:
                gt_lat, gt_lon = gt_map[p.stem]
                img = Image.open(p).convert("RGB")
                img_p = fn(img)
                pred_lat, pred_lon = run_plonk(pipe, img_p)
                err = haversine(gt_lat, gt_lon, pred_lat, pred_lon)
                errs.append(err)
            except Exception as e:
                print(f"⚠️ {variant} — erreur sur {p.name}: {e}")
        if errs:
            per_variant_errors[variant] = errs
            summary.append((variant, mean(errs), median(errs)))

    if not summary:
        print("⚠️ Aucune mesure d'erreur produite.")
        return

    summary.sort(key=lambda x: x[1])
    print("\n📊 Classement variantes (par moyenne d'erreur, km):")
    for name, m, med in summary:
        print(f" - {name:20s} → mean {m:8.2f} | median {med:7.2f} | n={len(per_variant_errors[name])}")

    best_name, best_mean, best_median = summary[0]
    print(f"\n🏆 Meilleure variante: {best_name} (mean {best_mean:.2f} km | median {best_median:.2f} km)")

    out_csv = Path("backend/plonk_diag_variants.csv")
    rows = []
    for var, errs in per_variant_errors.items():
        rows.append({"variant": var, "mean_km": f"{mean(errs):.3f}", "median_km": f"{median(errs):.3f}", "n": len(errs)})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","mean_km","median_km","n"])
        w.writeheader()
        w.writerows(rows)
    print(f"📂 Résumé variantes sauvegardé: {out_csv}")

if __name__ == "__main__":
    main()

