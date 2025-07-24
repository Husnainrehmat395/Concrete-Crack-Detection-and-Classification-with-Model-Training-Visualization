## This is the code for cropping and aligning the original image to comparison with another image during crack propagation analysis



import cv2
import numpy as np
import os
import re

def numeric_key(path):
    """
    Extract the integer in parentheses from filenames like 'beam(10).jpg'
    so we can sort images in true numeric order: (0), (1), (2), (10), etc.
    """
    filename = os.path.basename(path)
    match = re.search(r'\((\d+)\)', filename)
    return int(match.group(1)) if match else 999999

def register_image(base_img, target_img):
    """
    Register (align) target_img to base_img using feature matching, affine transformation or homography.
    Returns (aligned_image, H) where aligned_image is warped to base_img's coords.
    """
    # Convert to grayscale
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # SIFT detector (alternative to ORB for better robustness)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(base_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    # Match features using KNN
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # If not enough good matches, return original image
    if len(good_matches) < 10:
        print("Not enough good matches to compute homography.")
        return target_img, None

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography with RANSAC for better robustness
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        # If homography fails, fallback to affine transformation
        print("Homography could not be computed. Trying affine transformation...")
        M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts)
        if M is None:
            print("Affine transformation also failed. Returning original image.")
            return target_img, None
        # Apply affine transformation
        aligned = cv2.warpAffine(target_img, M, (base_img.shape[1], base_img.shape[0]))
        return aligned, M
    
    # If homography is successful, apply it to the image
    aligned = cv2.warpPerspective(target_img, H, (base_img.shape[1], base_img.shape[0]))
    return aligned, H

def main(folder_path, save_cropped=False, output_folder=None):
    """
    1) Sort images by numeric key.
    2) Let user define an ROI on the first image (the reference).
    3) For each subsequent image, register it to the reference, then crop the same ROI.
    4) Display or optionally save the cropped results.
    """
    # Gather and sort images
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(valid_ext)]
    if not image_files:
        print("No images found in:", folder_path)
        return
    image_files = sorted(image_files, key=numeric_key)

    # Read the reference image
    ref_path = image_files[0]
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print("Could not read reference image:", ref_path)
        return

    # Let the user draw an ROI on the reference image
    roi_window = "Select ROI on Reference - press ENTER or SPACE"
    r = cv2.selectROI(roi_window, ref_img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(roi_window)

    if r[2] == 0 or r[3] == 0:
        print("No ROI selected. Exiting.")
        return

    x_roi, y_roi, w_roi, h_roi = r
    print(f"Selected ROI on reference: x={x_roi}, y={y_roi}, w={w_roi}, h={h_roi}")

    # Optionally create output folder
    if save_cropped and output_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    # For each image in the folder
    for idx, fpath in enumerate(image_files):
        img = cv2.imread(fpath)
        if img is None:
            print("Could not read image:", fpath)
            continue

        filename_only = os.path.basename(fpath)

        
        if idx == 0:
            aligned_img = ref_img
        else:
            # Register to reference
            aligned_img, _ = register_image(ref_img, img)
            if aligned_img is None:
                print(f"Skipping {filename_only} (registration failed).")
                continue

        # Crop the same ROI
        H_aligned, W_aligned = aligned_img.shape[:2]
        # Check bounds
        if (x_roi + w_roi > W_aligned) or (y_roi + h_roi > H_aligned):
            print(f"ROI out of bounds for {filename_only}. Skipping.")
            continue

        cropped = aligned_img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

        # Show the cropped image with filename
        display = cropped.copy()
        cv2.putText(display, filename_only,
                    (10, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("Cropped Beam ROI", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

        # Optionally save
        if save_cropped and output_folder:
            out_name = f"cropped_{filename_only}"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, cropped)
            print(f"Saved cropped image: {out_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder = r"E:\C-1300-5.56"  
    # Example usage: main(folder, save_cropped=True, output_folder=r"E:\OutputCropped")
    main(folder, save_cropped=True, output_folder=r"E:\C-1300-5.56_Cropped")
