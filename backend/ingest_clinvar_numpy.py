#!/usr/bin/env python3
"""
Ingestion ClinVar VCF → NumPy compressé
Usage:
  python ingest_clinvar_numpy.py clinvar.vcf.gz output_dir
"""

import sys
import gzip
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python ingest_clinvar_numpy.py clinvar.vcf.gz output_dir")
    sys.exit(1)

vcf_file = sys.argv[1]
out_dir = sys.argv[2]

chroms = []
positions = []
refs = []
alts = []
signifs = []
ids = []

# Dictionnaire d'encodage clinique
signif_map = {
    "Pathogenic": 0,
    "Likely_pathogenic": 1,
    "Benign": 2,
    "Likely_benign": 3,
    "Uncertain_significance": 4,
}

def encode_significance(info_field):
    for key in signif_map:
        if key.lower() in info_field.lower():
            return signif_map[key]
    return 9  # inconnu

print(f"📥 Lecture {vcf_file} ...")
with gzip.open(vcf_file, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        chrom = fields[0]
        pos = int(fields[1])
        vid = fields[2] if fields[2] != "." else ""
        ref = fields[3]
        alt = fields[4]
        info = fields[7]

        chroms.append(chrom)
        positions.append(pos)
        refs.append(ref)
        alts.append(alt)
        ids.append(vid)
        signifs.append(encode_significance(info))

print(f"✅ {len(positions)} variants lus.")

chroms = np.array(chroms)
positions = np.array(positions, dtype=np.int32)
refs = np.array(refs)
alts = np.array(alts)
ids = np.array(ids)
signifs = np.array(signifs, dtype=np.int8)

np.savez_compressed(f"{out_dir}/clinvar_variants.npz",
    chrom=chroms,
    pos=positions,
    ref=refs,
    alt=alts,
    clinvar_id=ids,
    signif=signifs
)

print(f"💾 Sauvegardé dans {out_dir}/clinvar_variants.npz")

