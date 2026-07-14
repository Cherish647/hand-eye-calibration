import cv2
import numpy as np
import glob
import os
import math

# ====================== ⚠️ 1. 配置参数 (已为您修正为 8x6) ⚠️ ======================

IMAGE_DIR = r"D:\Eyehand\images"
POSE_FILE = r"D:\Eyehand\poses.txt"

# 🌟 修正：方块是 9x7，内角点就是 (9-1)x(7-1) = 8x6
CHESSBOARD_SIZE = (8, 6)
SQUARE_SIZE_MM = 20.0

CAMERA_MATRIX = np.array([
    [609.9689, 0, 640.7615],
    [0, 609.9598, 358.5411],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.array([-1.026230, 0.394129, -0.000509, -0.000630, 0.017902], dtype=np.float32)


# ====================== 数学辅助函数 ======================

def euler2rot(rx, ry, rz):
    x, y, z = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def read_robot_poses(filepath):
    poses = {}
    with open(filepath, 'r', encoding='gbk') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip() or line.startswith('#'): continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 7: continue
            img_name = parts[0]
            try:
                x, y, z, rx, ry, rz = map(float, parts[1:7])
                R_gripper2base = euler2rot(rx, ry, rz)
                t_gripper2base = np.array([[x], [y], [z]], dtype=np.float64)
                poses[img_name] = (R_gripper2base, t_gripper2base)
            except:
                continue
    return poses


# ====================== 核心流程 ======================

def main():
    print("🚀 启动【全算法加速】手眼标定流程...\n")

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    robot_poses = read_robot_poses(POSE_FILE)
    R_gripper2base_list, t_gripper2base_list = [], []
    R_target2cam_list, t_target2cam_list = [], []

    valid_images = 0
    images = glob.glob(os.path.join(IMAGE_DIR, '*.png')) + glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))

    for fname in images:
        img_name = os.path.basename(fname)
        if img_name not in robot_poses: continue

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- 🌟 阶段 1：尝试最新 Sector-Based (SB) 算法 (对无白边极其有效) ---
        # 这种算法在 OpenCV 4.5 以后版本默认集成，识别精度极高
        flags_sb = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        ret, corners = False, None
        try:
            ret, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, flags_sb)
        except AttributeError:
            pass  # 如果 OpenCV 版本太低则跳过

        # --- 🌟 阶段 2：如果 SB 算法失败，回退到增强型传统算法 ---
        if not ret:
            find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, find_flags)

            # 容错：尝试翻转长宽
            if not ret:
                ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_SIZE[1], CHESSBOARD_SIZE[0]), find_flags)

        if ret:
            # 亚像素级精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 解算 PnP
            _, rvec_cam, tvec_cam = cv2.solvePnP(objp, corners2, CAMERA_MATRIX, DIST_COEFFS)
            R_target2cam, _ = cv2.Rodrigues(rvec_cam)
            R_gripper2base, t_gripper2base = robot_poses[img_name]

            R_gripper2base_list.append(R_gripper2base)
            t_gripper2base_list.append(t_gripper2base)
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(tvec_cam)

            valid_images += 1
            print(f"✅ 成功处理: {img_name}")
        else:
            print(f"❌ 图像 {img_name} 识别失败。原因：标定板边缘模糊或参数不匹配。")

    print(f"\n📊 有效数据: {valid_images} 组")
    if valid_images < 3:
        print("❌ 错误：有效数据不足，无法解算。请确保标定板左右留出白边。")
        return

    print("⚙️ 正在进行矩阵解算 (AX=XB)...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    hand_eye_matrix = np.eye(4)
    hand_eye_matrix[:3, :3] = R_cam2gripper
    hand_eye_matrix[:3, 3] = t_cam2gripper.flatten()

    print("\n" + "=" * 60)
    print("🎉 标定结果：\n")
    print("HAND_EYE_MATRIX = np.array([")
    for i in range(4):
        print(
            f"    [{hand_eye_matrix[i, 0]:11.8f}, {hand_eye_matrix[i, 1]:11.8f}, {hand_eye_matrix[i, 2]:11.8f}, {hand_eye_matrix[i, 3]:11.8f}],")
    print("])")
    print("=" * 60)


if __name__ == "__main__":
    main()