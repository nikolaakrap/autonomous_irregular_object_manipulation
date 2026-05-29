import cv2
import numpy as np
import os
import socket
from ast import literal_eval
import pyrealsense2 as rs

SAVE_FOLDER = "data/calib_images"
POSITIONS_FILE = os.path.join(SAVE_FOLDER, "robot_positions.txt")
IMAGE_PREFIX = "image"
IMAGE_FORMAT = ".png"

ROBOT_HOST = "192.168.40.2" ### UPISATI IP ADRESU (PROVJERI JE LI 40.2 ILI 40.27)
ROBOT_PORT = 30002

def tcp_to_4x4(tcp):
    x, y, z, rx, ry, rz = tcp
    rvec = np.array([rx, ry, rz], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def save_4x4(filepath, matrix):
    with open(filepath, "a", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(f"{value:.8f}" for value in row) + "\n")
        f.write("\n")

def get_robot_tcp():
    server_socket = None
    client_socket = None
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((ROBOT_HOST, ROBOT_PORT))
        server_socket.listen(1)
        server_socket.settimeout(5.0)

        client_socket, addr = server_socket.accept()
        data = client_socket.recv(1024)
        tcp = literal_eval(data.decode("utf-8").strip())
        return tcp
    except Exception as e:
        print(f"[GREŠKA] Socket problem: {e}")
        return None
    finally:
        if client_socket: client_socket.close()
        if server_socket: server_socket.close()

os.makedirs(SAVE_FOLDER, exist_ok=True)
existing_images = [f for f in os.listdir(SAVE_FOLDER) if f.startswith(IMAGE_PREFIX) and f.endswith(IMAGE_FORMAT)]
counter = len(existing_images) + 1

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color)
pipeline.start(config)

print("\nSnimanje spremno. Pritisni SPACE za sliku, Q za izlaz.")
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue

        frame = np.asanyarray(color_frame.get_data())
        preview = frame.copy()
        cv2.putText(preview, f"Snimljeno: {counter - 1} | SPACE=Spremi | Q=Izlaz", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("Snimanje", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            img_path = os.path.join(SAVE_FOLDER, f"{IMAGE_PREFIX}_{counter:03d}{IMAGE_FORMAT}")
            cv2.imwrite(img_path, frame)
            
            tcp = get_robot_tcp()
            if tcp is None:
                os.remove(img_path)
                continue

            save_4x4(POSITIONS_FILE, tcp_to_4x4(tcp))
            print(f"[OK] Par #{counter} spremljen.")
            counter += 1
        elif key in (ord("q"), 27):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()