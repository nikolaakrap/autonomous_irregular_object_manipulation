import cv2
import numpy as np
import glob
from natsort import natsorted

IMAGES_GLOB = "data/calib_images/*.png"
ROBOT_FILE = "data/calib_images/robot_positions.txt"

SQUARE_LENGTH = 0.029
MARKER_LENGTH = 0.015
MIN_CHARUCO_CORNERS = 6

def load_4x4_matrices(filepath):
    matrices, current = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                if current: matrices.append(np.array(current, dtype=np.float64))
                current = []
            else: current.append([float(x) for x in line.split()])
    if current: matrices.append(np.array(current, dtype=np.float64))
    return matrices

def make_4x4(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3], T[:3, 3] = R, t.reshape(3)
    return T

def invert_4x4(T):
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3], T_inv[:3, 3] = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
    return T_inv

def detect_target_to_camera(image_path, mtx, dist, board, charuco_detector):
    img = cv2.imread(image_path)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS: return None
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
    return make_4x4(cv2.Rodrigues(rvec)[0], tvec) if success else None

mtx = np.load("data/camera_matrix.npy")
dist = np.load("data/dist_coeffs.npy")
board = cv2.aruco.CharucoBoard((5, 7), SQUARE_LENGTH, MARKER_LENGTH, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
charuco_detector = cv2.aruco.CharucoDetector(board)

image_paths = natsorted(glob.glob(IMAGES_GLOB))
robot_matrices = load_4x4_matrices(ROBOT_FILE)

R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = [], [], [], []

for i, img_path in enumerate(image_paths):
    if i >= len(robot_matrices): break
    T_target_cam = detect_target_to_camera(img_path, mtx, dist, board, charuco_detector)
    if T_target_cam is not None:
        R_gripper2base.append(robot_matrices[i][:3, :3])
        t_gripper2base.append(robot_matrices[i][:3, 3].reshape(3, 1))
        R_target2cam.append(T_target_cam[:3, :3])
        t_target2cam.append(T_target_cam[:3, 3].reshape(3, 1))

R_tcp2cam, t_tcp2cam = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_TSAI)
T_cam_from_tcp = make_4x4(R_tcp2cam, t_tcp2cam.reshape(3))
T_tcp_from_cam = invert_4x4(T_cam_from_tcp)

np.save("data/T_cam_from_tcp.npy", T_cam_from_tcp)
np.save("data/T_tcp_from_cam.npy", T_tcp_from_cam)
print("Spremljeno: data/T_cam_from_tcp.npy i data/T_tcp_from_cam.npy")