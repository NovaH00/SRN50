import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F


class SRN50_Triplet(nn.Module):
    def __init__(self, embedding_dim=128, freeze_backbone=True):
        super(SRN50_Triplet, self).__init__()
        
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
        
    def forward_once(self, x):
        
        features = self.backbone(x)    #Output: (batch_size, 2048, 1, 1)
        embedding = self.fc(features)  #Output: (batch_size, embedding_dim)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, anchor, positive, negative):

        embed_anchor   = self.forward_once(anchor)
        embed_positive = self.forward_once(positive)
        embed_negative = self.forward_once(negative)
        return embed_anchor, embed_positive, embed_negative
    

class SRN50_Constrative(nn.Module):
    def __init__(self, embedding_dim=128, freeze_backbone=True):
        super(SRN50_Constrative, self).__init__()
        
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1]) 
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
        
    def forward_once(self, x):
        features = self.backbone(x)
        embedding = self.fc(features)
        embedding = F.normalize(embedding, p=2, dim=1)  # Normalize embeddings
        return embedding

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)
