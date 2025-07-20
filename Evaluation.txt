#Unet_model
import cv2
import numpy as np
import torch
import os
import math
import re
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import segmentation_models_pytorch as smp

# =====================
# CONFIGURATION
# =====================
model_path = r"C:\Users\PMLS\Downloads\Linknet_model.pth"
class_names = {
    0: "Background",
    1: "Compression Crack",
    2: "Rebar Detachment",
    3: "Shear Type - 01",
    4: "Shear Type - 02",
    5: "Tension Crack"
}
CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0],
    5: [255, 0, 255]
}

# =====================
# LOAD MODEL
# =====================
device = torch.device("cpu")
model = smp.Linknet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=6
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# =====================
# GUI GLOBALS
# =====================
current_image_index = 0
image_files = []
original_img = None
pred_mask_resized = None
zoom_factor = 5
grid_size = 100
window_name = "Crack Detection"

# =====================
# FUNCTIONS
# =====================
def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_tensor = torch.tensor(image_resized.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    return image_rgb, image_tensor.to(device)

def predict_mask(image_tensor, orig_shape):
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return cv2.resize(pred_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

def draw_grid(img, size=100):
    h, w, _ = img.shape
    for x in range(0, w, size):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
    for y in range(0, h, size):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
    return img

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        col = x // grid_size
        row = y // grid_size
        x1, y1 = col * grid_size, row * grid_size
        x2, y2 = x1 + grid_size, y1 + grid_size
        cell_mask = pred_mask_resized[y1:y2, x1:x2]
        unique = np.unique(cell_mask)
        print(f"\n Clicked Cell ({row}, {col}) → Crack Types:")
        for class_id in unique:
            if class_id == 0:
                continue
            label = class_names.get(class_id, "Unknown")
            print(f"  - {label}")

def show_image():
    global original_img, pred_mask_resized
    image_path = image_files[current_image_index]
    print(f"\n🔍 Processing: {os.path.basename(image_path)}")
    original_img, image_tensor = preprocess_image(image_path)
    pred_mask_resized = predict_mask(image_tensor, original_img.shape[:2])
    color_mask = colorize_mask(pred_mask_resized)
    overlay = cv2.addWeighted(original_img, 0.7, color_mask, 0.3, 0)
    overlay_grid = draw_grid(overlay.copy(), size=grid_size)
    cv2.imshow(window_name, cv2.cvtColor(overlay_grid, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback(window_name, click_event)

# =====================
# GUI LOOP
# =====================
def main():
    global image_files, current_image_index

    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        messagebox.showerror("Error", "No folder selected.")
        return

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort(key=lambda f: natural_sort_key(os.path.basename(f)))

    if not image_files:
        messagebox.showerror("Error", "No image files found in the folder.")
        return

    cv2.namedWindow(window_name)
    show_image()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            current_image_index += 1
            if current_image_index < len(image_files):
                show_image()
            else:
                print("End of images.")
                break
        elif key == ord('b'):
            current_image_index = max(0, current_image_index - 1)
            show_image()
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
