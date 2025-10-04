#!/usr/bin/env python3
"""
Ingestion ClinVar VCF.gz → SQLite avec index optimisé
Usage:
    python backend/ingest_clinvar_sqlite.py clinvar.vcf.gz /Users/marco/fractus_db/clinvar_variants.db
"""

import sqlite3
import gzip
import sys
import os

def create_table(conn):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS variants (
        chrom TEXT,
        pos   INTEGER,
        ref   TEXT,
        alt   TEXT,
        clinvar_id TEXT,
        significance TEXT
    )
    """)
    # Index pour accélérer les recherches
    c.execute("CREATE INDEX IF NOT EXISTS idx_variants_chrom_pos ON variants (chrom, pos)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_variants_id ON variants (clinvar_id)")
    conn.commit()

def parse_and_insert(vcf_file, db_file, chunk_size=50000):
    if not os.path.exists(os.path.dirname(db_file)):
        os.makedirs(os.path.dirname(db_file))

    conn = sqlite3.connect(db_file)
    create_table(conn)
    c = conn.cursor()

    buffer = []
    total = 0

    with gzip.open(vcf_file, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.strip().split("\t")
            chrom, pos, clinvar_id, ref, alt, qual, flt, info = f[:8]

            # ClinVar encode la signification clinique dans INFO (champ CLNSIG)
            significance = "NA"
            for entry in info.split(";"):
                if entry.startswith("CLNSIG="):
                    significance = entry.split("=")[1]
                    break

            buffer.append((chrom, int(pos), ref, alt, clinvar_id, significance))

            if len(buffer) >= chunk_size:
                c.executemany("INSERT INTO variants VALUES (?, ?, ?, ?, ?, ?)", buffer)
                conn.commit()
                total += len(buffer)
                print(f"💾 {total} variants insérés...")
                buffer = []

    # dernier flush
    if buffer:
        c.executemany("INSERT INTO variants VALUES (?, ?, ?, ?, ?, ?)", buffer)
        conn.commit()
        total += len(buffer)

    conn.close()
    print(f"✅ Import terminé : {total} variants insérés dans {db_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ingest_clinvar_sqlite.py clinvar.vcf.gz /path/to/clinvar_variants.db")
        sys.exit(1)

    vcf_file = sys.argv[1]
    db_file = sys.argv[2]
    parse_and_insert(vcf_file, db_file)

