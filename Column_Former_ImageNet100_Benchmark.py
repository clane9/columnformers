#!/usr/bin/env python




# Import packages

# In[4]:


from torch.utils.data import DataLoader,Dataset
from datasets import load_dataset, load_from_disk
import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm

from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os

from columnformers.models.model_v1 import Columnformer, columnformer_v1_patch16_128
from columnformers.models import create_model
from columnformers.models.classification import ImageClassification
from torchsummary import summary

# Define gpus to use. For now only on one gpu

# In[5]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Importing the dataset from Hugging Face if not already available. If available, read from disk

# In[6]:


if os.path.isdir("../imagenet100.hf"):
    dataset = load_from_disk("../imagenet100.hf")
else:
    dataset = load_dataset("clane9/imagenet-100")
    dataset.save_to_disk("../imagenet100.hf")


# #### Preparing datasets

# In[7]:




# Define the type of training you will have.
# 
# "full" : Using the full ImageNet-100 for training. Not recommended on Colab unless you have a paid account
# 
# "debug" : Using a small subset of ImageNet-100 containing 6k images in training, sampled in a balanced manner from the 100 classes, used for debugging and testing the architecture. (AKA Micro-ImageNet-100)

# In[8]:


training_type = "full" # "debug" for a small 6k training subset of imagenet100, or "full" for the full imagenet100
batch_size = 512


# Paths of existing lists of labels to extract the Micro-ImageNet-100

# In[9]:


# Filenames for saving labels
train_labels_file = 'data/imagenet100/train_labels.txt'
test_labels_file = 'data/imagenet100/test_labels.txt'


# Extracting Micro-ImageNet-100

# In[10]:




# Function to load or extract labels
def load_or_extract_labels(dataset, labels_file):
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as file:
            return [int(line.strip()) for line in file.readlines()]
    else:
        labels = [dataset[i]['label'] for i in range(len(dataset))]
        with open(labels_file, 'w') as file:
            for label in labels:
                file.write(f'{label}\n')
        return labels


# In[11]:


# Define the custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}

# Updated transformations for the images
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Convert all images to RGB
    transforms.ColorJitter(0.4,0.4,0.2,0.1),
    transforms.RandomResizedCrop(size=(128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Convert all images to RGB
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Create instances of the custom dataset with transformations
train_dataset = CustomImageDataset(dataset['train'], transform=transform)
val_dataset = CustomImageDataset(dataset['validation'], transform=test_transform)


# In[12]:


if training_type == "debug" :
    # Load or extract labels for training and test datasets
    train_labels = load_or_extract_labels(dataset['train'], train_labels_file)
    test_labels = load_or_extract_labels(dataset['validation'], test_labels_file)


# In[13]:


if training_type == "debug" :
    # Stratified split
    sss_train = StratifiedShuffleSplit(n_splits=1, train_size=0.05, random_state=1)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=0)
    
    train_indices, _ = next(sss_train.split(np.zeros(len(train_labels)), train_labels))
    _, test_indices = next(sss_test.split(np.zeros(len(test_labels)), test_labels))
    
    # Convert indices to Python integers
    train_indices = [int(i) for i in train_indices]
    test_indices = [int(i) for i in test_indices]
    
    # Create subset datasets
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(val_dataset, test_indices)  # Use val_dataset for test subset
    
    # Create data loaders for subsets
    train_subset_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_subset_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


# In[14]:


if training_type == "debug" :
    # Assuming train_subset and test_subset are your final subset datasets
    
    train_subset_size = len(train_subset)
    test_subset_size = len(test_subset)
    
    print(f"Size of Training Subset: {train_subset_size}")
    print(f"Size of Testing Subset: {test_subset_size}")


# In[15]:


if training_type == "debug" :
    train_dataloader = train_subset_dataloader
    val_dataloader = test_subset_dataloader
elif training_type == "full" :
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


# ## Training the model
# 
# Importing the model architecture

# In[16]:



# In[17]:


torch.cuda.empty_cache()


# Initializing the model

# In[18]:


columnformer = create_model("columnformer_v1_patch16_128")
model = ImageClassification(
        encoder=columnformer,
        img_size=128,
        patch_size=16,
        output_len=256,
        num_classes=100,
        global_pool="avg",
    ).to(device)


# In[19]:


optimizer = optim.Adam(model.parameters(), lr=0.0001)


# In[20]:



print(summary(model))


# Training the model for 100 epochs

# In[ ]:


for epoch in range(100):  # 100 epochs
    model.train()  # Set the model to training mode
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm.tqdm(train_dataloader):
        optimizer.zero_grad()  # Zero the gradients

        # Move each tensor in the batch to the GPU
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass: Compute loss and state by passing the processed batch through the model
        loss, state = model({"image": images, "label": labels})

        outputs = state["output"]
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()
        total_loss += loss.item()

        # Optional: Print loss, you might want to accumulate and print every few iterations
    train_accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch}, Training Loss: {avg_loss}, Training Accuracy: {train_accuracy}%")


    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            loss, state = model({"image": images, "label": labels})
            val_loss += loss.item()
            outputs = state["output"]

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%")

