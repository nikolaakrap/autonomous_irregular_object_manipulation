import pyrealsense2 as rs
import numpy as np
import os

os.makedirs("data", exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color)
pipeline.start(config)

profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

print("--- Intrinzični parametri kamere ---")
print(f"Rezolucija: {intr.width} x {intr.height} piksela")
print(f"Fokalne duljine: fx = {intr.fx:.2f}, fy = {intr.fy:.2f} piksela")
print(f"Glavna točka: cx = {intr.ppx:.2f}, cy = {intr.ppy:.2f} piksela")
print(f"Distorzijski koeficijenti (k1, k2, p1, p2, k3): {intr.coeffs}")

mtx = np.array([
    [intr.fx, 0.0,     intr.ppx],
    [0.0,     intr.fy, intr.ppy],
    [0.0,     0.0,     1.0     ]
], dtype=np.float64)
dist = np.array(intr.coeffs, dtype=np.float64)

# Spremanje u novi data folder
np.save("data/camera_matrix.npy", mtx)
np.save("data/dist_coeffs.npy", dist)

print("\nIntrinsic matrix saved to: data/camera_matrix.npy")
print("Distortion coefficients saved to: data/dist_coeffs.npy")
pipeline.stop()