import random
from PIL import Image
import torchvision
from torch.utils.data import Dataset
import torchvision.utils
import torch.nn as nn

class TripletLossDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder  # ImageFolder instance
        self.transform = transform
        self.classes = folder.classes  # Store class names from ImageFolder

    def __getitem__(self, index):
        import random
        from PIL import Image

        # Randomly select an anchor image
        anchor_tuple = random.choice(self.folder.imgs)

        while True:
            positive_tuple = random.choice(self.folder.imgs)
            if anchor_tuple[1] == positive_tuple[1]:  # Same class
                break

        while True:
            negative_tuple = random.choice(self.folder.imgs)
            if anchor_tuple[1] != negative_tuple[1]:  # Different class
                break

        anchor = Image.open(anchor_tuple[0])
        positive = Image.open(positive_tuple[0])
        negative = Image.open(negative_tuple[0])

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.folder.imgs)


class ConstrativeLossDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset  # ImageFolder dataset
        self.transform = transform
        self.class_indices = self._create_class_indices()

    def _create_class_indices(self):
        """Creates a mapping of class labels to their image indices."""
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]

        if random.random() > 0.5:  # 50% chance of selecting positive pair
            idx2 = random.choice(self.class_indices[label1])  # Same class
            label = 1
        else:
            label2 = random.choice([l for l in self.class_indices.keys() if l != label1])
            idx2 = random.choice(self.class_indices[label2])  # Different class
            label = 0

        img2, _ = self.dataset[idx2]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label  # Output format: (image1, image2, label)
