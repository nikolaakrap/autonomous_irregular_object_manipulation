import cv2
import numpy as np
import glob
from natsort import natsorted
import os

# Putanje ažurirane prema data folderu
mtx = np.load("data/camera_matrix.npy")
dist = np.load("data/dist_coeffs.npy")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 7), 0.029, 0.015, aruco_dict)
charuco_detector = cv2.aruco.CharucoDetector(board)
MIN_CHARUCO_CORNERS = 6

images = natsorted(glob.glob("data/calib_images/*.png"))
os.makedirs("data/calib_results", exist_ok=True)
results_path = "data/calib_results/koordinate.txt"

with open(results_path, "w", encoding="utf-8") as f:
    for image_path in images:
        img = cv2.imread(image_path)
        if img is None: continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        display_img = img.copy()

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        if marker_ids is not None: cv2.aruco.drawDetectedMarkers(display_img, marker_corners, marker_ids)

        out_path = f"data/calib_results/{os.path.basename(image_path)}"

        if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
            cv2.imwrite(out_path, display_img)
            continue

        cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids)
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        success, rvec_charuco, tvec_charuco = cv2.solvePnP(obj_points, img_points, mtx, dist)

        if success:
            tx, ty, tz = tvec_charuco.flatten()
            rx, ry, rz = rvec_charuco.flatten()
            f.write(f"{os.path.basename(image_path)}: x={tx:.8f}, y={ty:.8f}, z={tz:.8f}, rx={rx:.8f}, ry={ry:.8f}, rz={rz:.8f}\n")
            cv2.drawFrameAxes(display_img, mtx, dist, rvec_charuco, tvec_charuco, 0.1)

        cv2.imwrite(out_path, display_img)
        cv2.imshow("Detection", display_img)
        cv2.waitKey(100)

cv2.destroyAllWindows()
print(f"\nGotovo. Rezultati zapisani u: data/calib_results/")