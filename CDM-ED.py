##############################################################################
                        # LIBRARY IMPORTS
##############################################################################

import cv2
import numpy as np
import math
import os
import re
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import filedialog

##############################################################################
                        # GLOBAL VARIABLES & CONSTANTS
##############################################################################

NUM_SCALING_IMAGES = 13  
image_files = []
current_image_index = 0
image = None
crack_image_on_bright = None
zoom_factor = 5  
is_zoomed = False
mouse_x, mouse_y = 0, 0
distance_mode = False
compute_cell_area_mode = False  
polygon_mode = False            
clicked_points = []             
polygon_points = []             
roi_x1, roi_y1 = 0, 0         
small_box_size_mm = 100  
scaling_factor_width = False
scaling_factor_height = False
distance_text = ""  
grid_spacing_mm = 100  
drag_start = None 
drag_end = None
highlighted_cell = None  

##############################################################################
                        # HELPER FUNCTIONS
##############################################################################

def draw_transparent_rectangle(img, x, y, w, h, color=(0,55,0), alpha=0.2):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def perspective_correction(img, contour):
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        pts = sorted(np.squeeze(approx), key=lambda x: (x[1], x[0]))
        pts1 = np.float32([pts[0], pts[1], pts[3], pts[2]])
        width, height = 100, 100
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, matrix, (width, height))
    return None

def detect_scaling_factors_in_image(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5,5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found_factors = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 0.9 < aspect_ratio < 1.1 and 1000 < area < 20000:
            corrected_box = perspective_correction(img_gray, contour)
            if corrected_box is not None:
                sfx = small_box_size_mm / w
                sfy = small_box_size_mm / h
                found_factors.append((sfx, sfy))
    return found_factors

##############################################################################
                        # FILENAME SORTING FUNCTION
##############################################################################

def numeric_key(path):
    filename = os.path.basename(path)
    match = re.search(r'\((\d+)\)', filename)
    return int(match.group(1)) if match else 999999

##############################################################################
                        # COMPUTE AVERAGE SCALING FACTOR
##############################################################################

def compute_average_scaling_factor_for_folder(folder_path):
    all_sfx = []
    all_sfy = []
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    all_files = sorted(all_files, key=numeric_key)
    num_images = min(NUM_SCALING_IMAGES, len(all_files))
    for img_path in all_files[:num_images]:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        factors = detect_scaling_factors_in_image(img)
        if factors:
            for (sfx, sfy) in factors:
                all_sfx.append(sfx)
                all_sfy.append(sfy)
    if len(all_sfx) == 0:
        return None, None
    avg_sfx = np.mean(all_sfx)
    avg_sfy = np.mean(all_sfy)
    return avg_sfx, avg_sfy

##############################################################################
                        # LOAD IMAGE FUNCTIONS
##############################################################################

def load_folder(folder_path):
    global image_files, current_image_index, scaling_factor_width, scaling_factor_height
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    image_files = sorted(image_files, key=numeric_key)
    if not image_files:
        messagebox.showerror("Error", "No image files found in the specified folder.")
        return
    avg_sfx, avg_sfy = compute_average_scaling_factor_for_folder(folder_path)
    if avg_sfx is not None:
        scaling_factor_width = avg_sfx
        scaling_factor_height = avg_sfy
        print(f"Global scaling factor (X): {scaling_factor_width:.2f} mm/px")
        print(f"Global scaling factor (Y): {scaling_factor_height:.2f} mm/px")
    else:
        scaling_factor_width = scaling_factor_height = None
        messagebox.showerror("Error", "No suitable scaling boxes found in the folder.")
    current_image_index = 0
    load_image()

def load_image():
    global image, crack_image_on_bright, highlighted_cell, polygon_points, polygon_mode
    if current_image_index >= len(image_files):
        messagebox.showinfo("End", "All images in the folder have been processed.")
        return
    image_path = image_files[current_image_index]
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", f"Image at path '{image_path}' could not be loaded.")
        return
    highlighted_cell = None
    polygon_points = []
    polygon_mode = False
    reset_main_image(img)

def reset_main_image(new_img):
    global image
    image = new_img
    process_image()

##############################################################a################
                        # IMAGE PROCESSING FUNCTIONS
##############################################################################

def remove_grid_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY_INV, 15, 10)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
    vertical_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel)
    grid_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)
    dilated_grid_mask = cv2.dilate(grid_mask, np.ones((3,3), np.uint8), iterations=2)
    img_no_grid = cv2.inpaint(img, dilated_grid_mask, 5, cv2.INPAINT_TELEA)
    return img_no_grid

def detect_cracks_no_area(image_no_grid):
    gray_image = cv2.cvtColor(image_no_grid, cv2.COLOR_BGR2GRAY)
    brightened_image = cv2.convertScaleAbs(gray_image, alpha=0.68, beta=12)
    _, binary_image = cv2.threshold(brightened_image, 85, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crack_image = image_no_grid.copy()
    cv2.drawContours(crack_image, contours, -1, (0,0,255), 2)
    return crack_image

def process_image():
    global image, crack_image_on_bright, scaling_factor_width, scaling_factor_height
    if current_image_index < NUM_SCALING_IMAGES:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if 0.9 < aspect_ratio < 1.1 and 1000 < area < 20000:
                corrected_box = perspective_correction(gray, contour)
                if corrected_box is not None:
                    draw_transparent_rectangle(image, x, y, w, h)
    if scaling_factor_width and scaling_factor_height:
        cv2.putText(image, f'Scaling Factor X: {scaling_factor_width:.2f} mm/px', (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(image, f'Scaling Factor Y: {scaling_factor_height:.2f} mm/px', (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    image_no_grid = remove_grid_lines(image)
    crack_image_on_bright = detect_cracks_no_area(image_no_grid)

##############################################################################
                    # CALCULATION & GRID FUNCTIONS
##############################################################################

def calculate_distance(point1, point2, scaling_factor_x, scaling_factor_y):
    dx = (point2[0] - point1[0]) * scaling_factor_x if scaling_factor_x else 0
    dy = (point2[1] - point1[1]) * scaling_factor_y if scaling_factor_y else 0
    return math.sqrt(dx**2 + dy**2)

def compute_polygon_area(points):
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.int32)
    area_px = cv2.contourArea(pts)
    if scaling_factor_width and scaling_factor_height:
        area_mm2 = area_px * (scaling_factor_width * scaling_factor_height)
        return area_mm2
    return area_px

def draw_grid_on_image(img, grid_spacing_x=50, grid_spacing_y=50):
    img_with_grid = img.copy()
    h, w = img_with_grid.shape[:2]
    for y in range(0, h, grid_spacing_y):
        cv2.line(img_with_grid, (0, y), (w, y), (0,0,0), 1)
    for x in range(0, w, grid_spacing_x):
        cv2.line(img_with_grid, (x, 0), (x, h), (0,0,0), 1)
    box_number = 1
    rows = h // grid_spacing_y
    cols = w // grid_spacing_x
    for row in range(rows):
        for col in range(cols):
            cell_x = col * grid_spacing_x
            cell_y = row * grid_spacing_y
            center_x = cell_x + grid_spacing_x // 2
            center_y = cell_y + grid_spacing_y // 2
            cv2.putText(img_with_grid, str(box_number),
                        (center_x - 10, center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            box_number += 1
    return img_with_grid

def compute_cell_area(x, y):
    global crack_image_on_bright, grid_spacing_mm, scaling_factor_width, scaling_factor_height, highlighted_cell
    src = crack_image_on_bright if crack_image_on_bright is not None else image
    if scaling_factor_width and scaling_factor_height:
        cell_width_px = int(grid_spacing_mm / scaling_factor_width)
        cell_height_px = int(grid_spacing_mm / scaling_factor_height)
    else:
        cell_width_px = cell_height_px = 50
    col = x // cell_width_px
    row = y // cell_width_px
    x1 = col * cell_width_px
    y1 = row * cell_height_px
    x2 = x1 + cell_width_px
    y2 = y1 + cell_height_px
    cell = src[y1:y2, x1:x2].copy()
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, binary_cell = cv2.threshold(gray_cell, 85, 255, cv2.THRESH_BINARY_INV)
    total_pixels = binary_cell.size
    white_pixels = cv2.countNonZero(binary_cell)
    black_pixels = total_pixels - white_pixels
    if scaling_factor_width and scaling_factor_height:
        px_to_mm2 = scaling_factor_width * scaling_factor_height
        crack_area_mm2 = white_pixels * px_to_mm2
        non_crack_area_mm2 = black_pixels * px_to_mm2
        crack_percent = (crack_area_mm2 / (total_pixels * px_to_mm2)) * 100 if total_pixels > 0 else 0
        non_crack_percent = 100 - crack_percent
        print(f"Cell ({row}, {col}):")
        print(f"  Crack Area (Black): {white_pixels} px, {crack_area_mm2:.2f} mm² ({crack_percent:.1f}%)")
        print(f"  Non-Crack Area (White): {black_pixels} px, {non_crack_area_mm2:.2f} mm² ({non_crack_percent:.1f}%)")
    else:
        crack_percent = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        non_crack_percent = 100 - crack_percent
        print(f"Cell ({row}, {col}):")
        print(f"  Crack Area (Black): {white_pixels} px ({crack_percent:.1f}%), Non-Crack Area (White): {black_pixels} px ({non_crack_percent:.1f}%)")
    highlighted_cell = (x1, y1, x2, y2)

##############################################################################
                        # MOUSE & MAIN LOOP FUNCTIONS
##############################################################################

def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, clicked_points, is_zoomed, drag_start, drag_end
    global distance_mode, distance_text, compute_cell_area_mode, polygon_mode, polygon_points, highlighted_cell

    if polygon_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            if is_zoomed:
                h_img, w_img = image.shape[:2]
                crop_w = int(w_img / zoom_factor)
                crop_h = int(h_img / zoom_factor)
                adjusted_x = roi_x1 + int(x * crop_w / w_img)
                adjusted_y = roi_y1 + int(y * crop_h / h_img)
            else:
                adjusted_x, adjusted_y = x, y
            polygon_points.append((adjusted_x, adjusted_y))
            cv2.circle(image, (adjusted_x, adjusted_y), 4, (255,0,0), -1)
            if len(polygon_points) > 1:
                cv2.line(image, polygon_points[-2], polygon_points[-1], (255,0,0), 2)
            cv2.imshow("Crack Detection", image)
            return
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(polygon_points) >= 3:
                cv2.line(image, polygon_points[-1], polygon_points[0], (255,0,0), 2)
                area = compute_polygon_area(polygon_points)
                text = f"Poly Area: {area:.2f} mm^2" if (scaling_factor_width and scaling_factor_height) else f"Poly Area: {area:.2f} px"
                print(text)
                cv2.putText(image, text, (polygon_points[0][0], polygon_points[0][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
                cv2.imshow("Crack Detection", image)
            polygon_mode = False
            polygon_points = []
            return

    if compute_cell_area_mode and event == cv2.EVENT_LBUTTONDOWN:
        compute_cell_area(x, y)
        return

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
        if drag_start:
            drag_end = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        if is_zoomed:
            h_img, w_img = image.shape[:2]
            crop_w = int(w_img / zoom_factor)
            crop_h = int(h_img / zoom_factor)
            adjusted_x = roi_x1 + int(x * crop_w / w_img)
            adjusted_y = roi_y1 + int(y * crop_h / h_img)
        else:
            adjusted_x, adjusted_y = x, y

        if distance_mode:
            if len(clicked_points) == 2:
                clicked_points.clear()
                idx_img = cv2.imread(image_files[current_image_index])
                if idx_img is not None:
                    reset_main_image(idx_img)
            clicked_points.append((adjusted_x, adjusted_y))
            if len(clicked_points) == 1:
                cv2.circle(image, (adjusted_x, adjusted_y), 5, (0,0,255), -1)
            elif len(clicked_points) == 2:
                cv2.circle(image, (adjusted_x, adjusted_y), 5, (0,0,255), -1)
                cv2.line(image, clicked_points[0], (adjusted_x, adjusted_y), (0,255,0),2)
                dist = calculate_distance(clicked_points[0], (adjusted_x, adjusted_y),
                                          scaling_factor_width, scaling_factor_height)
                distance_text = f"Distance: {dist:.2f} mm"
            cv2.imshow("Crack Detection", image)
            return

        if drag_start is None:
            drag_start = (x, y)
            drag_end = None

    elif event == cv2.EVENT_LBUTTONUP:
        if drag_start:
            x1, y1 = drag_start
            x2, y2 = x, y
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            if scaling_factor_width and scaling_factor_height:
                area_pixels = width * height
                area_mm2 = area_pixels * scaling_factor_width * scaling_factor_height
                print(f"Selected Area: {area_mm2:.2f} mm²")
            else:
                print("Selected Area: (No scaling factor)")
            drag_start = None

def main_loop():
    global image, crack_image_on_bright, current_image_index, is_zoomed, distance_mode
    global roi_x1, roi_y1, distance_text, compute_cell_area_mode, highlighted_cell, grid_spacing_mm
    global polygon_mode, polygon_points

    if image is None:
        messagebox.showerror("Error", "No image loaded. Please check the folder path.")
        return

    cv2.namedWindow("Crack Detection")
    cv2.setMouseCallback("Crack Detection", on_mouse)

    while True:
        h, w = image.shape[:2]
        if crack_image_on_bright is not None:
            base_display = cv2.addWeighted(image, 0.7, crack_image_on_bright, 0.3, 0)
        else:
            base_display = image.copy()

        if is_zoomed:
            crop_w = int(w / zoom_factor)
            crop_h = int(h / zoom_factor)
            zoom_x1 = max(0, min(mouse_x - crop_w // 2, w - crop_w))
            zoom_y1 = max(0, min(mouse_y - crop_h // 2, h - crop_h))
            roi_x1, roi_y1 = zoom_x1, zoom_y1
            if scaling_factor_width and scaling_factor_height:
                grid_spacing_x = int(grid_spacing_mm / scaling_factor_width)
                grid_spacing_y = int(grid_spacing_mm / scaling_factor_height)
            else:
                grid_spacing_x = grid_spacing_y = 80

            full_grid = draw_grid_on_image(image,
                                           grid_spacing_x=grid_spacing_x,
                                           grid_spacing_y=grid_spacing_y)
            grid_crop = full_grid[zoom_y1:zoom_y1+crop_h, zoom_x1:zoom_x1+crop_w]
            grid_resized = cv2.resize(grid_crop, (w, h), interpolation=cv2.INTER_LINEAR)
            crop = base_display[zoom_y1:zoom_y1+crop_h, zoom_x1:zoom_x1+crop_w]
            display_image = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            display_image = cv2.addWeighted(display_image, 0.9, grid_resized, 0.1, 0)
            if highlighted_cell is not None:
                hx1, hy1, hx2, hy2 = highlighted_cell
                new_hx1 = int((hx1 - roi_x1) * w / crop_w)
                new_hy1 = int((hy1 - roi_y1) * h / crop_h)
                new_hx2 = int((hx2 - roi_x1) * w / crop_w)
                new_hy2 = int((hy2 - roi_y1) * h / crop_h)
                cv2.rectangle(display_image, (new_hx1, new_hy1), (new_hx2, new_hy2), (0,0,255), 3)
        else:
            display_image = base_display.copy()
            if highlighted_cell is not None:
                cv2.rectangle(display_image,
                              (highlighted_cell[0], highlighted_cell[1]),
                              (highlighted_cell[2], highlighted_cell[3]),
                              (0,0,255), 3)
            if scaling_factor_width and scaling_factor_height:
                grid_spacing_x = int(grid_spacing_mm / scaling_factor_width)
                grid_spacing_y = int(grid_spacing_mm / scaling_factor_height)
            else:
                grid_spacing_x = grid_spacing_y = 80
            display_image = draw_grid_on_image(display_image,
                                               grid_spacing_x=grid_spacing_x,
                                               grid_spacing_y=grid_spacing_y)

        if drag_start and drag_end:
            x1, y1 = drag_start
            x2, y2 = drag_end
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0,255,255),2)
        if distance_text:
            text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = display_image.shape[1] - text_size[0] - 20
            text_y = display_image.shape[0] - 20
            cv2.putText(display_image, distance_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2, cv2.LINE_AA)
        if polygon_points:
            for i in range(len(polygon_points)-1):
                cv2.line(display_image, polygon_points[i], polygon_points[i+1], (255,0,0),2)
            for pt in polygon_points:
                cv2.circle(display_image, pt, 4, (255,0,0), -1)

        # Displaying the image name dynamically
        filename_only = os.path.basename(image_files[current_image_index]) if current_image_index < len(image_files) else "No image"
        cv2.setWindowTitle("Crack Detection", "Crack Detection - " + filename_only)

        cv2.imshow("Crack Detection", display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            current_image_index += 1
            if current_image_index < len(image_files):
                load_image()
            else:
                print("End of images in folder.")
        elif key == ord('b'):
            current_image_index = max(0, current_image_index - 1)
            load_image()
        elif key == ord('m'):
            is_zoomed = not is_zoomed
        elif key == ord('d'):
            distance_mode = not distance_mode
        elif key == ord('c'):
            compute_cell_area_mode = not compute_cell_area_mode
            if not compute_cell_area_mode:
                highlighted_cell = None
            print("Cell Area Mode ON" if compute_cell_area_mode else "Cell Area Mode OFF")
        elif key == ord('p'):
            polygon_mode = not polygon_mode
            if polygon_mode:
                polygon_points = []
                print("Polygon mode ON: Left-click to add vertices, right-click to finish.")
            else:
                print("Polygon mode OFF.")
        elif key == ord('g'):
            new_spacing = simpledialog.askfloat("Grid Spacing", "Enter grid spacing in mm:", initialvalue=grid_spacing_mm)
            if new_spacing is not None and new_spacing > 0:
                grid_spacing_mm = new_spacing
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

##############################################################################
                                # MAIN BLOCK
##############################################################################

def select_folder():
    folder_selected = filedialog.askdirectory(title="Select Folder")
    if folder_selected:
        print(f"Selected folder: {folder_selected}")
        return folder_selected
    else:
        print("No folder selected")
        return None

folder_path = select_folder() 
if folder_path:
    load_folder(folder_path)
    main_loop()
else:
    messagebox.showerror("Error", "No folder selected.")
    
