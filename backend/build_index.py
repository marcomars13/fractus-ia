# ~/Projets/fractus-ia/backend/build_index.py

import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Fichiers d'entrée / sortie
CSV_FILE = "mapillary_out/images.csv"
OUTPUT_EMB = "mapillary_out/embeddings.npy"
OUTPUT_META = "mapillary_out/meta.csv"
OUTPUT_INDEX = "mapillary_out/mapillary.faiss"

# Charger CSV (doit contenir colonnes filepath, lat, lon)
df = pd.read_csv(CSV_FILE)
assert {"filepath", "lat", "lon"}.issubset(df.columns), \
    f"CSV {CSV_FILE} doit contenir filepath, lat, lon"

# Charger StreetCLIP
MODEL_ID = "geolocal/StreetCLIP"
print(f"🚀 Chargement modèle {MODEL_ID}...")
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()

embeddings = []
meta = []

print(f"📸 Encodage de {len(df)} images...")
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        image = Image.open(row["filepath"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feat = model.get_image_features(**inputs).cpu().numpy().ravel()
        embeddings.append(feat)
        meta.append([row["filepath"], row["lat"], row["lon"]])
    except Exception as e:
        print(f"⚠️ Erreur image {row['filepath']}: {e}")

# Sauvegarde embeddings et meta
embeddings = np.array(embeddings, dtype="float32")
np.save(OUTPUT_EMB, embeddings)
pd.DataFrame(meta, columns=["filepath","lat","lon"]).to_csv(OUTPUT_META, index=False)

# Construire index FAISS
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
faiss.write_index(index, OUTPUT_INDEX)

print(f"✅ Embeddings sauvegardés: {OUTPUT_EMB}")
print(f"✅ Meta sauvegardées: {OUTPUT_META}")
print(f"✅ Index FAISS sauvegardé: {OUTPUT_INDEX}")

