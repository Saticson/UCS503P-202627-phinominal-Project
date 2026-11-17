"""
Sukio / AxiomCloud - Computer Vision Training, Inference & Grad-CAM XAI Engine
Genuine PyTorch training, validation, serialization, reload, and real Grad-CAM heatmaps.
"""

import os
import sys
import json
import time
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ConvNet(nn.Module):
    """Deep convolutional neural network with accessible feature map hooks for Grad-CAM."""
    def __init__(self, num_classes=2, in_channels=3):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),  # 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),  # 7x7
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleImageDataset(Dataset):
    def __init__(self, image_data, labels, transform=None):
        self.image_data = image_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img = self.image_data[idx]
        if isinstance(img, (list, tuple)):
            img = np.array(img, dtype=np.uint8)
        if isinstance(img, str):
            if os.path.exists(img):
                img = Image.open(img).convert('RGB')
            else:
                img = Image.open(io.BytesIO(base64.b64decode(img))).convert('RGB')
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img).convert('RGB')
        elif not isinstance(img, Image.Image):
            raise ValueError(f"Unsupported image type: {type(img)}")

        img = img.resize((28, 28))
        img_np = np.array(img, dtype=np.float32) / 255.0
        # Normalization (0.5 mean, 0.5 std)
        img_tensor = torch.tensor(img_np).permute(2, 0, 1)
        img_tensor = (img_tensor - 0.5) / 0.5
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label

class GradCAM:
    """Calculates true Gradient-weighted Class Activation Mapping for ConvNet."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Channel-wise mean of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)  # ReLU
        
        # Upsample to input tensor size
        cam = torch.nn.functional.interpolate(cam, size=(28, 28), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Min-max normalization
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, torch.softmax(output, dim=1).detach().cpu().numpy()[0]

def train_vision_model(image_data, labels, class_names, num_epochs=5, batch_size=32, lr=0.001, test_size=0.2):
    """Trains a genuine PyTorch vision model, validates on held-out split, and saves model artifact."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = len(labels)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split = int(n_samples * (1 - test_size))
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = SimpleImageDataset([image_data[i] for i in train_idx], [labels[i] for i in train_idx])
    val_dataset = SimpleImageDataset([image_data[i] for i in val_idx], [labels[i] for i in val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(class_names)
    model = ConvNet(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time

    # Evaluate on held-out validation set
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    accuracy = correct / max(1, total)
    val_loss = val_loss / max(1, total)

    return {
        "model": model,
        "class_names": class_names,
        "val_accuracy": round(accuracy, 4),
        "val_loss": round(val_loss, 4),
        "train_time_sec": round(train_time, 2),
        "samples_evaluated": total
    }

def save_vision_artifact(model, class_names, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "num_classes": len(class_names),
        "class_names": class_names,
        "architecture": "ConvNet"
    }
    torch.save(checkpoint, output_path)
    return output_path

def load_vision_artifact(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = ConvNet(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["class_names"]

if __name__ == "__main__":
    print("Testing genuine PyTorch Vision Trainer and Grad-CAM...")
    # Synthetic test raster to verify architecture and Grad-CAM backprop
    test_imgs = [np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8) for _ in range(40)]
    test_labels = [0 if i < 20 else 1 for i in range(40)]
    res = train_vision_model(test_imgs, test_labels, ["Class_0", "Class_1"], num_epochs=2)
    print(f"Vision Training Success! Val Accuracy: {res['val_accuracy']}, Train Time: {res['train_time_sec']}s")
    
    # Test Grad-CAM
    model = res["model"]
    gradcam = GradCAM(model, model.features[-1])  # Hook on last Conv2d
    sample_tensor = torch.randn(1, 3, 28, 28)
    heatmap, pred_class, probs = gradcam.generate_heatmap(sample_tensor)
    print(f"Grad-CAM Success! Pred Class: {pred_class}, Heatmap Shape: {heatmap.shape}, Max Intensity: {heatmap.max():.4f}")
