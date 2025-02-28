import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader
from siamese_models import SiameseResNet50

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Model
model = SiameseResNet50(embedding_dim=256).to(device)
model.load_state_dict(torch.load("siamese2.pth", map_location=device, weights_only=True))
model.eval()

# Define Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet normalization
])

# Load Test Data
test_dataset = datasets.ImageFolder(r"data/capsule/train", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Choose an anchor image
anchor_image = Image.open(r"data/capsule-small/test/good/021.png")
anchor_image = transform(anchor_image).unsqueeze(0).to(device)

# Define Distance Functions
def cos_sim(embed1, embed2):
    return torch.nn.functional.cosine_similarity(embed1, embed2).item()

def euclidean_dist(embed1, embed2):
    return torch.norm(embed1 - embed2, p=2).item()

print(test_dataset.class_to_idx)

THRESHOLD = 0.9998 


true_labels = []
predicted_labels = []
from collections import Counter
with torch.no_grad():
    anchor_embed = model.forward_once(anchor_image)  # Get anchor embedding
    
    for image, label in test_loader:
        image, label = image.to(device), label.item()  # Move to device
        output_embed = model.forward_once(image)

        # Compute Cosine Similarity
        similarity = cos_sim(anchor_embed, output_embed)
        euclid_dist = euclidean_dist(anchor_embed, output_embed)
        # # Predict Same (1) or Different (0) Class
        # predicted_label = 1 if similarity > THRESHOLD else 0

        # true_labels.append(label)
        # predicted_labels.append(predicted_label)
        
        # print(Counter(true_labels))
        
        print(similarity, euclid_dist)



# # Compute Metrics
# accuracy = accuracy_score(true_labels, predicted_labels)
# precision = precision_score(true_labels, predicted_labels, average='macro')
# recall = recall_score(true_labels, predicted_labels, average='macro')
# f1 = f1_score(true_labels, predicted_labels, average='macro')

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-score: {f1:.4f}")

# # Confusion Matrix
# cm = confusion_matrix(true_labels, predicted_labels)

# # Plot Confusion Matrix
# plt.figure(figsize=(5, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
