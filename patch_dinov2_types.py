import os
import re

# dossier DINOv2 à patcher
dinov2_dir = "/Users/marco/.cache/torch/hub/facebookresearch_dinov2_main/dinov2"

pattern = re.compile(r"(\w+)\s*\|\s*None")

for root, _, files in os.walk(dinov2_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                code = f.read()
            # ajoute import Optional si besoin
            if pattern.search(code):
                if "from typing import Optional" not in code:
                    code = "from typing import Optional\n" + code
                code = pattern.sub(r"Optional[\1]", code)
                with open(path, "w") as f:
                    f.write(code)
                print("✅ Patched:", path)

