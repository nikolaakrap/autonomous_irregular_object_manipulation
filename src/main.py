import os
import cv2
import time
import socket
import struct
import subprocess
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from ultralytics import YOLO

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
RESULTS_DIR = Path("results")
TRAJ_DIR    = RESULTS_DIR / "trajectories"
CAPTURE_DIR = Path("capture_data")

for d in [RESULTS_DIR, TRAJ_DIR, CAPTURE_DIR / "rgb", CAPTURE_DIR / "depth", CAPTURE_DIR / "pose"]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH   = "data/best.pt"
TARGET_CLASS = "orah"
ROBOT_IP     = "192.168.40.27"
PORT_30003   = 30003
PORT_SCRIPT  = 30001

try:
    T_tcp_from_cam = np.load("data/T_cam_from_tcp.npy").astype(np.float64)
except FileNotFoundError:
    print("[ERROR] T_cam_from_tcp.npy nije pronađen u data/ folderu!")
    exit(1)

HOME_POZA = np.array([-0.22993, 0.27334, 0.20202, 3.041, -0.880, -0.060])
PLACE_POZA = np.array([-0.25896, 0.19872, 0.02600, 3.174, -0.525, -0.068]) 

POZE_SNIMANJA = [
    np.array([-0.05255, 0.28205, 0.28217, 3.322, -0.303, -0.453]),
    np.array([-0.08089, 0.26299, 0.26057, 3.263, -0.032, 0.262]),
    np.array([-0.07884, 0.30087, 0.25966, 3.063, -1.452, 0.136])
]

DEPTH_MIN = 0.05
DEPTH_MAX = 1.00
APPROACH_Z_OFFSET = 0.10
TRAJ_DT, TRAJ_V_MAX, TRAJ_A_MAX = 0.008, 0.15, 0.3

# =============================================================================
# ROBOT CONTROL FUNCTIONS
# =============================================================================
def zatrazi_pozu_30003(ip=ROBOT_IP):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        s.connect((ip, PORT_30003))
        return np.array(struct.unpack('!6d', s.recv(2048)[444:492]))

def send_urscript_and_wait(script_body, ip=ROBOT_IP, port=PORT_SCRIPT, timeout=300.0):
    full_script = "def prog():\n" + "".join([f"  {l}\n" for l in script_body.splitlines()]) + '  textmsg("TRAJ_DONE")\nend\n'
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
        sock.sendall(full_script.encode("utf-8"))
        buf = ""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data = sock.recv(4096).decode("utf-8", errors="ignore")
                if data and "TRAJ_DONE" in (buf := buf + data): return
            except socket.timeout: break
        raise TimeoutError("[URSCRIPT] Nema TRAJ_DONE potvrde!")
    finally: sock.close()

def send_trajectory_file(filepath):
    with open(filepath, "r") as f: lines = [l.rstrip() for l in f.readlines() if l.strip()]
    send_urscript_and_wait("\n".join(lines))

def send_gripper_command(do_open):
    send_urscript_and_wait(f"set_digital_out(5, True)\nsleep(1.0)\nset_digital_out(5, False)" if do_open else f"set_digital_out(4, True)\nsleep(1.0)\nset_digital_out(4, False)")

def quintic_s(tau): return 10*(tau**3) - 15*(tau**4) + 6*(tau**5)

def generate_segment(p_i, p_f, R_i, R_f, dt=TRAJ_DT):
    p_i, p_f = np.asarray(p_i, dtype=np.float64), np.asarray(p_f, dtype=np.float64)
    D = float(np.linalg.norm(p_f - p_i))
    t_f = 0.5 if D < 1e-6 else max(D / TRAJ_V_MAX, np.sqrt(D / TRAJ_A_MAX), 0.3)
    t = np.arange(0.0, t_f, dt)
    s = quintic_s(t / t_f if t_f > 0 else np.zeros_like(t))
    p_path = np.tile(p_i, (len(t), 1)) if D < 1e-6 else p_i[None, :] + (p_f - p_i)[None, :] * s[:, None]
    rotvec = Slerp([0.0, 1.0], R.concatenate([R_i, R_f]))(s).as_rotvec()
    return np.hstack([p_path, rotvec])

def save_task_space_trajectory(traj, filepath):
    with open(filepath, "w") as f:
        for r in traj: f.write(f"servoj(get_inverse_kin(p[{r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}, {r[3]:.6f}, {r[4]:.6f}, {r[5]:.6f}]), t={TRAJ_DT:.4f}, lookahead_time=0.1, gain=300)\n")
        f.write("stopj(1.0)\n")

def tcp_to_matrix(tcp):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3], _ = cv2.Rodrigues(np.array(tcp[3:6], dtype=np.float64))
    T[:3, 3] = tcp[:3]
    return T

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("Učitavam YOLO model i kameru...")
    model = YOLO(MODEL_PATH)
    
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)

    print("\n[INIT] Idem u HOME pozu...")
    trenutna_poza = zatrazi_pozu_30003()
    traj = generate_segment(trenutna_poza[:3], HOME_POZA[:3], R.from_rotvec(trenutna_poza[3:6]), R.from_rotvec(HOME_POZA[3:6]))
    save_task_space_trajectory(traj, str(TRAJ_DIR / "init_home.txt"))
    send_trajectory_file(str(TRAJ_DIR / "init_home.txt"))
    trenutna_poza = HOME_POZA

    sve_tocke_oraha_baza = []

    # --- 1. RAW DATA CAPTURE ---
    for i, cilj in enumerate(POZE_SNIMANJA):
        print(f"\n[SNIMANJE] Idem na pozu {i+1}...")
        save_task_space_trajectory(generate_segment(trenutna_poza[:3], cilj[:3], R.from_rotvec(trenutna_poza[3:6]), R.from_rotvec(cilj[3:6])), str(TRAJ_DIR / f"pos_{i}.txt"))
        send_trajectory_file(str(TRAJ_DIR / f"pos_{i}.txt"))
        trenutna_poza = cilj

        time.sleep(1.5)
        for _ in range(15): pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames(timeout_ms=5000))
        rgb = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * frames.get_depth_frame().get_units()
        tcp_aktualni = zatrazi_pozu_30003()

        ts = int(time.time()*1000)
        cv2.imwrite(str(CAPTURE_DIR / "rgb" / f"rgb_{i+1}_{ts}.png"), rgb)
        np.save(str(CAPTURE_DIR / "depth" / f"depth_{i+1}_{ts}.npy"), depth)
        np.save(str(CAPTURE_DIR / "pose" / f"pose_{i+1}_{ts}.npy"), tcp_aktualni)

        T_base_from_cam = tcp_to_matrix(tcp_aktualni) @ T_tcp_from_cam
        results = model.predict(rgb, conf=0.15, verbose=False)[0]

        if results.masks is not None:
            for j, cls_id in enumerate(results.boxes.cls):
                if TARGET_CLASS in model.names[int(cls_id.item())].lower():
                    mraw = results.masks.data[j].cpu().numpy()
                    mask = cv2.resize(mraw, (640, 480), interpolation=cv2.INTER_NEAREST) > 0.5
                    rows, cols = np.where(mask)
                    z = depth[rows, cols]
                    
                    valid = (z > DEPTH_MIN) & (z < DEPTH_MAX)
                    rows, cols, z = rows[valid], cols[valid], z[valid]
                    
                    if len(z) > 20:
                        pts_cam = np.stack([(cols - K[0,2])*z/K[0,0], (rows - K[1,2])*z/K[1,1], z], axis=-1)
                        pts_base = (T_base_from_cam @ np.hstack([pts_cam, np.ones((len(pts_cam), 1))]).T).T[:, :3]
                        sve_tocke_oraha_baza.append(pts_base)
                        print(f" -> Detektirano {len(pts_base)} sirovih točaka oraha.")

                        pcd_temp = o3d.geometry.PointCloud()
                        pcd_temp.points = o3d.utility.Vector3dVector(pts_base)
                        o3d.io.write_point_cloud(str(RESULTS_DIR / f"PC_{i+1}_segmentirano.pcd"), pcd_temp)
                        break

    print("\n[POVRATAK] Idem u HOME...")
    save_task_space_trajectory(generate_segment(trenutna_poza[:3], HOME_POZA[:3], R.from_rotvec(trenutna_poza[3:6]), R.from_rotvec(HOME_POZA[3:6])), str(TRAJ_DIR / "home.txt"))
    send_trajectory_file(str(TRAJ_DIR / "home.txt"))
    pipeline.stop()

    if len(sve_tocke_oraha_baza) == 0:
        print("[GREŠKA] Orah nije detektiran! Izlazim.")
        return

    # --- 2. C++ COMM ---
    print("\n[C++ INTEGRACIJA] Spremamo sirovi oblak za C++...")
    pcd_sirovi = o3d.geometry.PointCloud()
    pcd_sirovi.points = o3d.utility.Vector3dVector(np.vstack(sve_tocke_oraha_baza))
    pcd_path = str(RESULTS_DIR.absolute() / "sirovi_oblak_baza.pcd")
    
    o3d.io.write_point_cloud(pcd_path, pcd_sirovi, write_ascii=True)

    print(f"Pokrećem C++ i šaljem mu točnu putanju: {pcd_path}")
    cpp_exe = os.path.abspath("./src/02_vision/build/vision_pipeline")
    
    subprocess.run([cpp_exe, pcd_path], cwd=str(RESULTS_DIR.absolute()))

    try:
        with open(RESULTS_DIR / "tocan_centar_oraha.txt", "r") as f:
            c = f.read().strip().split()
            c_sphere_f = np.array([float(c[0]), float(c[1]), float(c[2])])
    except Exception as e:
        print(f"[GREŠKA] Ne mogu pročitati rezultat od C++: {e}")
        return

    print(f"\n[REZULTAT] C++ je izračunao centar: X={c_sphere_f[0]:.4f}, Y={c_sphere_f[1]:.4f}, Z={c_sphere_f[2]:.4f}")

    # --- 3. PICK AND PLACE ---
    print("\n================ PICK & PLACE ================\n")
    PICK = np.array([c_sphere_f[0], c_sphere_f[1], c_sphere_f[2], HOME_POZA[3], HOME_POZA[4], HOME_POZA[5]])
    APP_PICK, APP_PLACE = PICK.copy(), PLACE_POZA.copy()
    APP_PICK[2] += APPROACH_Z_OFFSET; APP_PLACE[2] += APPROACH_Z_OFFSET

    def rot(p): return R.from_rotvec(p[3:6])
    seg_list = [
        ("app_pick", HOME_POZA, APP_PICK), 
        ("pick", APP_PICK, PICK), 
        ("lift", PICK, APP_PICK),
        ("move_to_place", APP_PICK, APP_PLACE),
        ("place", APP_PLACE, PLACE_POZA), 
        ("lift_place", PLACE_POZA, APP_PLACE), 
        ("home_final", APP_PLACE, HOME_POZA)
    ]

    for n, p_i, p_f in seg_list: 
        save_task_space_trajectory(generate_segment(p_i[:3], p_f[:3], rot(p_i), rot(p_f)), str(TRAJ_DIR / f"{n}.txt"))

    send_trajectory_file(str(TRAJ_DIR / "app_pick.txt"))
    send_trajectory_file(str(TRAJ_DIR / "pick.txt"))
    send_gripper_command(False)
    
    send_trajectory_file(str(TRAJ_DIR / "lift.txt"))
    send_trajectory_file(str(TRAJ_DIR / "move_to_place.txt"))
    send_trajectory_file(str(TRAJ_DIR / "place.txt"))
    
    send_gripper_command(True)
    send_trajectory_file(str(TRAJ_DIR / "lift_place.txt"))
    send_trajectory_file(str(TRAJ_DIR / "home_final.txt"))

    print("\nPROCESS FINISHED!\n")

if __name__ == "__main__":
    main()