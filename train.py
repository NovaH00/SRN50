from siamese_models import SRN50_Constrative
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from custom_datasets import ConstrativeLossDataset
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from tqdm import tqdm  # Progress bar


# 🔹 Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


# 🔹 Load Dataset
train_images = datasets.ImageFolder(r"data\augmented_dataset", transform=None)
train_dataset = ConstrativeLossDataset(train_images, transform=transform)


# 🔹 Model, Loss, Optimizer
model = SRN50_Constrative(embedding_dim=128)
contrastive_loss = nn.CosineEmbeddingLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 🔹 Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 🔹 Learning Rate Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250, eta_min=1e-6)


# 🔹 Function: Adjust Batch Size Dynamically
def adjust_batch_size(epoch, initial_batch_size=8, max_batch_size=64, step=5):
    """Gradually increases batch size every 'step' epochs."""
    return min(initial_batch_size * (2 ** (epoch // step)), max_batch_size)



# 🔹 Training Loop
num_epochs = 250
for epoch in range(num_epochs):
    total_loss = 0
    
    # 🔹 Update batch size
    batch_size = adjust_batch_size(epoch, max_batch_size=32)
    
    # 🔹 Recreate DataLoader with new batch size
    batch_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=batch_size, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for step, (image1, image2, label) in enumerate(train_loader):
        # Move to device
        image1, image2, label = image1.to(device), image2.to(device), label.to(device)
        
        optimizer.zero_grad()  # Reset gradients
        
        # Forward pass
        output1, output2 = model(image1, image2)
        loss = contrastive_loss(output1, output2, label.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 🔹 Progress Bar Update
        progress_bar.update(1)
        progress_bar.set_postfix(batch_loss=loss.item(), avg_loss=total_loss/(step+1), refresh=True)
    
    # 🔹 Update Scheduler
    scheduler.step()

    print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Batch size: {batch_size}")

# 🔹 Save Model
torch.save(model.state_dict(), "model/siamese-contrastive.pth")
