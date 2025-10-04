import numpy as np
import fractus_core

test_img_path = "/Users/marco/mapillary_france_adaptive/thumbs/1000091355214658.jpg"

print("🚀 Test 1 : liste simple")
print(fractus_core.run_fractus_full_api([0.1, 0.2, 0.3]))

print("\n🚀 Test 2 : np.ndarray 1D")
print(fractus_core.run_fractus_full_api(np.array([0.1, 0.2, 0.3], dtype=np.float32)))

print("\n🚀 Test 3 : np.ndarray 2D (fake image)")
print(fractus_core.run_fractus_full_api(np.random.rand(64, 64).astype(np.float32)))

print("\n🚀 Test 4 : np.ndarray 3D (fake RGB image)")
print(fractus_core.run_fractus_full_api(np.random.rand(64, 64, 3).astype(np.float32)))

print("\n🚀 Test 5 : dict avec chemin")
print(fractus_core.run_fractus_full_api({"image": test_img_path}))

