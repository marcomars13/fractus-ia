# backend/check_fractus_core.py
import fractus_core

print("🔎 Fonctions et attributs publics dans fractus_core :")
for name in dir(fractus_core):
    if not name.startswith("_"):
        print(" -", name)

