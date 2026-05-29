# Autonomous Irregular Object Manipulation

This repository contains a perception-manipulation system for the autonomous pick-and-place of irregular objects using a robotic arm. The system utilizes advanced computer vision, deep learning, and point cloud processing to locate and manipulate objects. 

<video src="https://github.com/user-attachments/assets/e94872fd-3d52-4aa9-938e-5fe25d84e984" ></video>

## Model Training: YOLOv26nano & 3D Printed Fruit
To enable the primary extraction of target objects from the background workspace, an Instance Segmentation model based on the **YOLOv26nano** architecture was trained. 
Unlike standard bounding boxes, the model outputs a dense pixel matrix (mask) that precisely describes the 2D contours of the object. 
The training dataset consisted of various **3D printed fruits**. After the instance segmentation model was successfully trained on this dataset, the best performing weights were saved as `best.pt` and are located in the `data/` directory.

### RGB Camera Views
Before generating the point cloud, the robot moves to three predefined spatial recording poses to capture RGB-D images of the 3D printed fruit.
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/771a2cf9-b376-447a-adea-10ae993777c4" />
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/21406a71-8feb-4270-854b-45a4ca472b7c" />
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/43674fe4-3dc5-48b6-805a-71937be99c1b" />

## System Architecture
The data acquisition, machine vision, and robot communication modules are implemented in Python, while the heavy processing and filtering of point clouds are implemented in C++. The workflow follows an 8-step pipeline:

1. **Homing & Initialization:** Reading the robot pose via Port 30003.
2. **Acquisition:** RGB-D triggering using an Intel RealSense D435 camera.
3. **YOLOv26nano:** 2D masking of the target object.
4. **3D Reconstruction:** Transformation of points from the camera frame to the robot base frame.
5. **C++ PCL Module:** Statistical Outlier Removal (SOR) filtering and geometric table cutting.
6. **Least Squares Fitting:** Calculating the sphere centroid using the Eigen library.
7. **Trajectory Planning:** Motion profiling using a 5th-degree polynomial approximation.
8. **Execution:** Sending serial servo commands to the robot via Port 30001.

## 3D Point Cloud Processing
Once the 2D masks are generated from the three different camera angles, the depth pixels are projected into 3D space. The data is processed through several stages:

* **Raw Cloud:** The segmented and registered point cloud is projected directly from the YOLO 2D masks.
<img width="957" height="537" alt="Image" src="https://github.com/user-attachments/assets/5a31a091-9830-48db-b167-456d4a3ffa5b" />
<img width="957" height="537" alt="Image" src="https://github.com/user-attachments/assets/ee356e9f-a747-408f-945f-396b7ebdf5d7" />

* **Filtered Cloud (Noise Removal):** A Statistical Outlier Removal (SOR) filter is applied to remove noise caused by depth camera errors. For each point, the algorithm finds the k-nearest neighbors and discards points exceeding the allowed tolerance (Parameters used: MeanK = 20, StddevMulThresh = 0.8).
<img width="957" height="537" alt="Image" src="https://github.com/user-attachments/assets/da601a02-067c-4851-9250-abcfe1ba460d" />

* **Final Cloud (Geometric Cutting):** To isolate the top dome of the object, geometric cutting is performed by sorting the Z-coordinates and removing the strict bottom 40% of the point distribution (Cutoff ratio = 0.40).
<img width="957" height="537" alt="Image" src="https://github.com/user-attachments/assets/b829f81c-afc2-4d6a-8f35-2c10a2d3b035" />

## Object Detection & Centroid Estimation
The cropped 3D point cloud is passed to the Eigen library, where the object is approximated by Least Squares sphere fitting. 
The system uses SVD decomposition to solve the linear system and find the exact spatial center of the sphere, which directly serves as the Target Center Point (TCP) coordinate for the robot's pick action.
<img width="957" height="537" alt="Image" src="https://github.com/user-attachments/assets/47a3a774-fcee-4546-9a24-73c9de6dd271" />

## Trajectory Planning & Analysis
Task Space motion planning is handled by approximating the curve with a 5th-degree polynomial (Quintic S-curve), guaranteeing continuous acceleration without jerks. The sequence includes safely approaching the pick location, grasping the object, moving to the place zone, and returning home.
<img width="1000" height="800" alt="Image" src="https://github.com/nikolaakrap/autonomous_irregular_object_manipulation/blob/main/media/pcl02.png" />
<img width="1000" height="1000" alt="Image" src="https://github.com/user-attachments/assets/97b519c7-59ef-4c23-a22c-b1089fe83898" />

## Setup & Build Instructions

### 1. Change Robot IP Address
**IMPORTANT:** Before running any scripts, ensure the robot's IP address matches your network setup. You need to update the IP address variables in the following files:
* `src/01_calibration/02_skupljanje_podataka.py` (Update ROBOT_HOST)
* `src/main.py` (Update ROBOT_IP)

### 2. Build the C++ Vision Pipeline
The heavy point cloud processing module is written in C++ and needs to be compiled using CMake. From the root directory of the repository, run the following commands:

    cd src/02_vision
    mkdir build
    cd build
    cmake ..
    make

## Running the System

### Step 1: Hand-Eye Calibration
To correctly map the 3D camera space to the robot's base coordinate system, execute the calibration scripts located in the `src/01_calibration/` directory sequentially:

1. `python 01_intrinzicni.py`: Extracts the camera's intrinsic parameters and saves `camera_matrix.npy` and `dist_coeffs.npy` to the `data/` directory.
2. `python 02_skupljanje_podataka.py`: Captures robot TCP poses and RGB images of the ChArUco calibration board, saving them to the `data/calib_images/` folder.
3. `python 03_eks_koord_sustavi.py`: Detects the ChArUco corners from the images and saves the spatial point coordinates to `data/calib_results/koordinate.txt`.
4. `python 04_hand_eye.py`: Computes the final transformation matrices using the Tsai-Lenz method and saves the `T_cam_from_tcp.npy` and `T_tcp_from_cam.npy` matrices to the `data/` directory.

### Step 2: Main Pick-and-Place Execution
Once the calibration is complete and the `data/` directory contains all the necessary `.npy` matrices alongside your YOLO `best.pt` model, the system is ready. You only need to run the main script:

    python src/main.py

This single script will automatically establish TCP/IP communication with the robot, load the YOLO model, capture multi-angle images, internally call the compiled `vision_pipeline` C++ executable for 3D processing, generate a smooth trajectory, and execute the physical manipulation task.
