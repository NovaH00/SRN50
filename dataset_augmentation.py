import os
import torch
import random
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # Progress bar

# 🔹 Input and Output Dataset Paths
input_root = "data/capsule/test"         # Your original dataset folder
output_root = "data/augmented_dataset"  # New folder where augmented images will be saved
num_augmented = 5  # Number of augmented copies per image

# 🔹 Define Augmentations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

# 🔹 Loop Through Each Class Folder and Augment Images
for class_name in os.listdir(input_root):
    class_path = os.path.join(input_root, class_name)
    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        img_path = os.path.join(class_path, img_name)
        image = Image.open(img_path).convert("RGB")

        # Save the original image to the new dataset
        image.save(os.path.join(output_class_path, img_name))

        # Generate augmented images
        for i in range(num_augmented):
            augmented_image = augmentation_transforms(image)
            new_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            augmented_image.save(os.path.join(output_class_path, new_name))

print("✅ Dataset Augmentation Complete! Check 'augmented_dataset' folder.")
