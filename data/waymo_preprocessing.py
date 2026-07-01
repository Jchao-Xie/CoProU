import tensorflow as tf
import os
import io
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset
import datetime
import pytz
import numpy as np
import cv2
import csv


def undistort_image(image_np, K, dist):
    h, w = image_np.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    return cv2.undistort(image_np, K, dist, None, new_K)

def extract_front_camera_and_pose(tfrecord_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    pose_file = os.path.join(output_dir, "poses.txt")
    intrinsic_file = os.path.join(output_dir, "intrinsics.txt")
    extrinsic_file = os.path.join(output_dir, "extrinsic_T_vehicle_camera.txt")

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    intrinsic_matrix = None
    extrinsic_matrix = None

    target_x = 1920
    target_y = 1280
    scale_x = target_x / 1920
    scale_y = target_y / 1280

    with open(pose_file, 'w') as f_pose:
        for i, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(data.numpy())

            # Load front camera image
            front_image_np = None
            for img in frame.images:
                if img.name == open_dataset.CameraName.FRONT:
                    image = Image.open(io.BytesIO(img.image))
                    front_image_np = np.array(image)  # shape: (H, W, 3)
                    break

            if front_image_np is None:
                continue

            # Get calibration for FRONT camera
            for cam in frame.context.camera_calibrations:
                if cam.name == open_dataset.CameraName.FRONT:
                    # Parse intrinsics
                    fx, fy, cx, cy = cam.intrinsic[:4]
                    k1, k2, p1, p2, k3 = cam.intrinsic[4:9]
                    full_K = np.array([[fx, 0, cx],
                                       [0, fy, cy],
                                       [0,  0,  1]])
                    dist = np.array([k1, k2, p1, p2, k3])

                    # Undistort image
                    front_image_np = undistort_image(front_image_np, full_K, dist)

                    # Resize to target size
                    if front_image_np.shape[0] != target_y or front_image_np.shape[1] !=target_x:
                        front_image_np = cv2.resize(front_image_np, (target_x, target_y))

                    # Save resized + undistorted image
                    cv2.imwrite(
                        os.path.join(output_dir, "images", f"frame_{i:04d}.jpg"),
                        cv2.cvtColor(front_image_np, cv2.COLOR_RGB2BGR)
                    )

                    # Scale intrinsics for resized image
                    fx *= scale_x
                    fy *= scale_y
                    cx *= scale_x
                    cy *= scale_y
                    K = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0,  0,  1]])

                    if intrinsic_matrix is None:
                        intrinsic_matrix = K
                        np.savetxt(intrinsic_file, intrinsic_matrix, fmt="%.6f")
                        print("📷 Saved scaled intrinsic matrix to", intrinsic_file)
                    else:
                        if not np.allclose(K, intrinsic_matrix, atol=1e-6):
                            print(f"❗ Intrinsic matrix changed at frame {i}!")
                            print(K)

                    # Get extrinsic (T_vehicle_from_camera)
                    T_vehicle_from_camera = np.array(cam.extrinsic.transform).reshape(4, 4)
                    if extrinsic_matrix is None:
                        extrinsic_matrix = T_vehicle_from_camera
                        np.savetxt(extrinsic_file, extrinsic_matrix, fmt="%.6f")
                        print("📍 Saved camera extrinsic to", extrinsic_file)
                    else:
                        if not np.allclose(T_vehicle_from_camera, extrinsic_matrix, atol=1e-6):
                            print(f"❗ Extrinsic matrix changed at frame {i}!")
                            print(T_vehicle_from_camera)
                    break

            # Compute T_world_camera = T_world_vehicle @ T_vehicle_camera
            T_world_vehicle = np.array(frame.pose.transform).reshape(4, 4)
            T_world_camera = T_world_vehicle @ T_vehicle_from_camera
            T_3x4 = T_world_camera[:3, :]
            f_pose.write(' '.join(f"{v:.6f}" for v in T_3x4.flatten()) + '\n')

    print(f"✅ Extraction complete: {i+1} frames processed into '{output_dir}'")
    

def get_day_segments(tfrecord_files, data_dir, out_root,
                     brightness_threshold=50):
    """
    Filter Waymo TFRecords into usable (day+dawn+dusk+rainy) vs. night scenes
    using masked mean brightness of the first FRONT camera image.

    Saves split lists (sequence names only) and first-frame previews.

    Args:
        tfrecord_files (list[str]): List of .tfrecord filenames
        data_dir (str): Directory containing .tfrecord files
        out_root (str): Output root directory
        brightness_threshold (float): Brightness cutoff for day/night
    """
    os.makedirs(out_root, exist_ok=True)
    day_dir = os.path.join(out_root, "preview_day")
    night_dir = os.path.join(out_root, "preview_night")
    os.makedirs(day_dir, exist_ok=True)
    os.makedirs(night_dir, exist_ok=True)

    out_day = os.path.join(out_root, "day_split.txt")
    out_night = os.path.join(out_root, "night_split.txt")
    out_csv = os.path.join(out_root, "brightness_summary.csv")

    day_segments, night_segments = [], []
    brightness_stats = []

    for tf_name in tfrecord_files:
        tf_path = os.path.join(data_dir, tf_name)
        dataset = tf.data.TFRecordDataset(tf_path, compression_type='')

        for data in dataset.take(1):  # only first frame
            frame = open_dataset.Frame()
            frame.ParseFromString(data.numpy())

            # Decode FRONT camera image
            front_img = None
            for img in frame.images:
                if img.name == open_dataset.CameraName.FRONT:
                    front_img = tf.image.decode_jpeg(img.image).numpy()
                    break

            if front_img is None:
                print(f"[WARN] No FRONT image found for {tf_name}")
                continue

            # Compute masked mean brightness (ignore top 5% brightest pixels)
            gray = cv2.cvtColor(front_img, cv2.COLOR_RGB2GRAY)
            gray_flat = gray.flatten()
            cutoff = int(0.95 * len(gray_flat))
            masked_mean = float(np.mean(np.sort(gray_flat)[:cutoff]))

            # Clean sequence name (remove extension)
            seq_name = os.path.splitext(tf_name)[0]

            brightness_stats.append((seq_name, masked_mean))

            # Save image preview
            out_name = seq_name + ".jpg"
            if masked_mean > brightness_threshold:
                day_segments.append(seq_name)
                save_path = os.path.join(day_dir, out_name)
            else:
                night_segments.append(seq_name)
                save_path = os.path.join(night_dir, out_name)

            cv2.imwrite(save_path, cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
            print(f"[INFO] {seq_name}: masked brightness={masked_mean:.1f}")

    # Save split lists (sequence names only)
    with open(out_day, "w") as f:
        f.writelines([seg + "\n" for seg in day_segments])
    with open(out_night, "w") as f:
        f.writelines([seg + "\n" for seg in night_segments])

    # Save brightness summary CSV
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sequence_name", "masked_mean_brightness"])
        writer.writerows(brightness_stats)

    print(f"[INFO] Found {len(day_segments)} usable (day+dawn+dusk+rainy) segments.")
    print(f"[INFO] Found {len(night_segments)} night segments.")
    print(f"[INFO] Saved brightness summary to {out_csv}")
    print(f"[INFO] Preview images stored in:\n  {day_dir}\n  {night_dir}")
    return day_segments, night_segments, brightness_stats

# 🔁 Loop over all .tfrecord files in a directory
if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing .tfrecord files")
    parser.add_argument("--out_root", required=True, help="Root output directory")
    parser.add_argument("--split-idx", required=True, type=int, help="split index")
    parser.add_argument("--split-num", required=True, type=int, help="total number of splits")
    parser.add_argument("--mode", default='processing', choices=['processing', 'get_day_sequence'], help="Select mode: process tfrecords or extract day sequences")
    args = parser.parse_args()

    tfrecord_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".tfrecord")])
    start_idx = len(tfrecord_files) * args.split_idx // args.split_num
    end_idx = len(tfrecord_files) * (args.split_idx + 1) // args.split_num

    
    if args.mode == 'processing':
        tfrecord_files = tfrecord_files[start_idx:end_idx]
        for tfrecord_file in tqdm(tfrecord_files):
            tfrecord_path = os.path.join(args.data_dir, tfrecord_file)
            segment_name = os.path.splitext(tfrecord_file)[0]
            output_dir = os.path.join(args.out_root, segment_name)

            print(f"📂 Processing {tfrecord_file} ...")
            extract_front_camera_and_pose(tfrecord_path, output_dir)
        
    elif args.mode == 'get_day_sequence':
        get_day_segments(tfrecord_files, args.data_dir, args.out_root)
