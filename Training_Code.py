import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from pycocotools.coco import COCO
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score

# Setup dataset path (update to your local path)
dataset_path = r"C:\CrackDetection"

# Define dice score function
def dice_score(preds, targets, smooth=1e-6):
    # Apply softmax and get the prediction of class with the highest score
    preds = torch.argmax(preds, dim=1)
    intersection = (preds & targets).float().sum((1, 2))  # Element-wise intersection
    union = (preds | targets).float().sum((1, 2))  # Element-wise union
    dice = (2. * intersection + smooth) / (union + intersection + smooth)  # Dice formula
    return dice.mean().item()  # Mean Dice score over the batch

# Dataset class
class CrackDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            cat_id = ann['category_id']
            mask_tmp = self.coco.annToMask(ann)
            mask[mask_tmp == 1] = cat_id

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask

# Transformations
transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])

val_transform = A.Resize(512, 512)

# Prepare datasets and dataloaders
train_dataset = CrackDataset(f'{dataset_path}\\train\\images', f'{dataset_path}\\train\\_annotations.coco.json', transform=transform)
val_dataset = CrackDataset(f'{dataset_path}\\valid\\images', f'{dataset_path}\\valid\\_annotations.coco.json', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")  # This will show if it's using the GPU or CPU

# Define the model
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    classes=6,    # 0=bg, 1-5 for your 5 crack types
    activation=None
).to(device)  # Move model to GPU if available

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5
train_loss_list, val_loss_list = [], []
train_dice_list, val_dice_list = [], []
val_acc_list, val_prec_list, val_rec_list, val_f1_list, val_iou_list = [], [], [], [], []

for epoch in range(num_epochs):
    model.train()
    train_loss, train_dice = 0.0, 0.0

    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_dice += dice_score(outputs.cpu(), masks.cpu())  # Moving to CPU for score calculation

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss, val_dice = 0.0, 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)  # Move data to GPU
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_dice += dice_score(outputs.cpu(), masks.cpu())  # Moving to CPU for score calculation

            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            targets = masks.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    # Compute additional metrics
    val_acc = accuracy_score(all_targets, all_preds)
    val_prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    val_rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    val_iou = jaccard_score(all_targets, all_preds, average='macro', zero_division=0)

    # Append to lists for later plotting
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_dice_list.append(train_dice)
    val_dice_list.append(val_dice)
    val_acc_list.append(val_acc)
    val_prec_list.append(val_prec)
    val_rec_list.append(val_rec)
    val_f1_list.append(val_f1)
    val_iou_list.append(val_iou)

    # Print summary for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    print(f"Train Dice: {train_dice:.4f} - Val Dice: {val_dice:.4f}")
    print(f"Val Acc: {val_acc:.4f} - Val Prec: {val_prec:.4f} - Val Rec: {val_rec:.4f} - Val F1: {val_f1:.4f} - Val IoU: {val_iou:.4f}")

# Save the model
torch.save(model.state_dict(), "crack_detection_model_gpu.pth")
print(" Model saved.")

# Save the history of training
import json
history = {
    "train_loss": train_loss_list,
    "val_loss": val_loss_list,
    "train_dice": train_dice_list,
    "val_dice": val_dice_list,
    "val_acc": val_acc_list,
    "val_prec": val_prec_list,
    "val_rec": val_rec_list,
    "val_f1": val_f1_list,
    "val_iou": val_iou_list
}
with open("training_history.json", "w") as f:
    json.dump(history, f)
print("Training history saved locally.")
