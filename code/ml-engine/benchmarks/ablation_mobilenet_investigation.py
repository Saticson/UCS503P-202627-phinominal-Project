"""
Sukio / AxiomCloud - Validation-Only Controlled Ablation Investigation for MobileNet-V3
Investigates root causes of underperformance on BreastMNIST, RetinaMNIST, and PneumoniaMNIST:
1. Spatial resolution: 28x28 vs 112x112 vs 224x224
2. Input channel handling: Weight projection vs 3-channel replication vs Random init
3. Normalization: [-1, 1] vs ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. Freeze/Unfreeze: Full fine-tune vs Linear probe (frozen backbone) vs Two-stage
5. Class weighting: None vs Inverse-frequency vs Sqrt-frequency
6. Learning rate: 1e-4 vs 3e-4 vs 1e-3

Strict Rule: Evaluated EXCLUSIVELY on Validation Split (test split is NEVER observed during ablation).
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score
import torchvision.models as models

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "datasets", "real", "vision"))

def build_ablation_model(num_classes, in_channels, channel_strategy="project", freeze_backbone=False):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    base = models.mobilenet_v3_small(weights=weights)
    
    if in_channels == 1:
        if channel_strategy == "project":
            orig_conv = base.features[0][0]
            new_conv = nn.Conv2d(1, orig_conv.out_channels, kernel_size=orig_conv.kernel_size,
                                 stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
            base.features[0][0] = new_conv
        elif channel_strategy == "random":
            orig_conv = base.features[0][0]
            base.features[0][0] = nn.Conv2d(1, orig_conv.out_channels, kernel_size=orig_conv.kernel_size,
                                            stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
        elif channel_strategy == "replicate":
            # Keep base as 3 channels, input will be replicated
            pass

    if freeze_backbone:
        for p in base.features.parameters():
            p.requires_grad = False

    in_features = base.classifier[0].in_features
    base.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(0.20),
        nn.Linear(128, num_classes)
    )
    return base

def run_ablation_trial(data_name, resolution=112, channel_strategy="project",
                       norm_type="imagenet", freeze_backbone=False,
                       class_weight_type="none", lr=3e-4, epochs=3):
    np.random.seed(42)
    torch.manual_seed(42)
    
    path = os.path.join(DATA_DIR, f"{data_name}.npz")
    data = np.load(path)
    tr_x, tr_y = data["train_images"], data["train_labels"].squeeze()
    va_x, va_y = data["val_images"], data["val_labels"].squeeze()
    
    # Cap training samples for fast ablation iteration
    if len(tr_x) > 1000:
        tr_x, tr_y = tr_x[:1000], tr_y[:1000]
    
    num_classes = len(np.unique(tr_y))
    in_channels = 1 if len(tr_x.shape) == 3 else tr_x.shape[-1]
    
    # 1. Channel formatting
    if in_channels == 1 and channel_strategy == "replicate":
        tr_x = np.repeat(tr_x[:, None, :, :], 3, axis=1).astype(np.float32) / 255.0
        va_x = np.repeat(va_x[:, None, :, :], 3, axis=1).astype(np.float32) / 255.0
        actual_in_channels = 3
    elif in_channels == 1:
        tr_x = tr_x[:, None, :, :].astype(np.float32) / 255.0
        va_x = va_x[:, None, :, :].astype(np.float32) / 255.0
        actual_in_channels = 1
    else:
        tr_x = np.transpose(tr_x.astype(np.float32) / 255.0, (0, 3, 1, 2))
        va_x = np.transpose(va_x.astype(np.float32) / 255.0, (0, 3, 1, 2))
        actual_in_channels = 3

    # 2. Normalization
    if norm_type == "imagenet":
        if actual_in_channels == 3:
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            tr_x = (tr_x - mean) / std
            va_x = (va_x - mean) / std
        else:
            tr_x = (tr_x - 0.456) / 0.224
            va_x = (va_x - 0.456) / 0.224
    else:
        tr_x = (tr_x - 0.5) / 0.5
        va_x = (va_x - 0.5) / 0.5

    # 3. Resizing
    t_tr_x = torch.tensor(tr_x, dtype=torch.float32)
    t_va_x = torch.tensor(va_x, dtype=torch.float32)
    if resolution != 28:
        t_tr_x = nn.functional.interpolate(t_tr_x, size=(resolution, resolution), mode='bilinear', align_corners=False)
        t_va_x = nn.functional.interpolate(t_va_x, size=(resolution, resolution), mode='bilinear', align_corners=False)

    t_tr_y = torch.tensor(tr_y, dtype=torch.long)
    t_va_y = torch.tensor(va_y, dtype=torch.long)

    # 4. Class weighting
    counts = np.bincount(tr_y, minlength=num_classes)
    if class_weight_type == "inverse":
        w = torch.tensor([len(tr_y) / (num_classes * max(1, c)) for c in counts], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=w)
    elif class_weight_type == "sqrt":
        w = torch.tensor([np.sqrt(len(tr_y) / max(1, c)) for c in counts], dtype=torch.float32)
        w = w / w.mean()
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    model = build_ablation_model(num_classes, actual_in_channels, channel_strategy, freeze_backbone)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(t_tr_x, t_tr_y), batch_size=32, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None
    
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vo = model(t_va_x)
            v_loss = criterion(vo, t_va_y).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        v_probs = torch.softmax(model(t_va_x), dim=1).numpy()
        v_preds = np.argmax(v_probs, axis=1)

    val_bal_acc = balanced_accuracy_score(va_y, v_preds)
    val_macro_f1 = f1_score(va_y, v_preds, average='macro', zero_division=0)
    
    return {
        "val_bal_acc": round(float(val_bal_acc), 4),
        "val_macro_f1": round(float(val_macro_f1), 4),
        "val_loss": round(float(best_val_loss), 4),
        "pred_counts": np.bincount(v_preds, minlength=num_classes).tolist()
    }

def main():
    print("=" * 80)
    print("CONTROLLED VALIDATION-ONLY ABLATIONS (BREASTMNIST, RETINAMNIST, PNEUMONIAMNIST)")
    print("=" * 80)

    # 1. BreastMNIST Ablations
    print("\n--- [Ablation: BreastMNIST] ---")
    trials_breast = [
        ("Baseline (28x28, [-1,1], project, inverse weights, lr=3e-4)", dict(resolution=28, norm_type="tanh", class_weight_type="inverse")),
        ("Upsample 112x112 + ImageNet Norm + Project weights + Unweighted", dict(resolution=112, norm_type="imagenet", class_weight_type="none")),
        ("Upsample 112x112 + ImageNet Norm + Replicate 3-ch + Unweighted", dict(resolution=112, norm_type="imagenet", channel_strategy="replicate", class_weight_type="none")),
        ("Upsample 112x112 + ImageNet Norm + Replicate 3-ch + Sqrt-weighted", dict(resolution=112, norm_type="imagenet", channel_strategy="replicate", class_weight_type="sqrt")),
        ("Upsample 112x112 + ImageNet Norm + Linear Probe (Frozen)", dict(resolution=112, norm_type="imagenet", channel_strategy="replicate", freeze_backbone=True, class_weight_type="none", lr=1e-3)),
        ("Upsample 224x224 + ImageNet Norm + Replicate 3-ch + Unweighted", dict(resolution=224, norm_type="imagenet", channel_strategy="replicate", class_weight_type="none"))
    ]
    for name, params in trials_breast:
        res = run_ablation_trial("breastmnist", **params)
        print(f"  {name}:")
        print(f"    --> Val Bal Acc: {res['val_bal_acc']:.4f} | Val F1: {res['val_macro_f1']:.4f} | Pred Counts: {res['pred_counts']}")

    # 2. RetinaMNIST Ablations
    print("\n--- [Ablation: RetinaMNIST] ---")
    trials_retina = [
        ("28x28 + [-1,1] + Inverse weights", dict(resolution=28, norm_type="tanh", class_weight_type="inverse")),
        ("112x112 + ImageNet Norm + Unweighted + lr=3e-4", dict(resolution=112, norm_type="imagenet", class_weight_type="none", lr=3e-4)),
        ("112x112 + ImageNet Norm + Sqrt-weighted + lr=3e-4", dict(resolution=112, norm_type="imagenet", class_weight_type="sqrt", lr=3e-4)),
        ("112x112 + ImageNet Norm + Linear Probe (Frozen) + lr=1e-3", dict(resolution=112, norm_type="imagenet", freeze_backbone=True, class_weight_type="none", lr=1e-3))
    ]
    for name, params in trials_retina:
        res = run_ablation_trial("retinamnist", **params)
        print(f"  {name}:")
        print(f"    --> Val Bal Acc: {res['val_bal_acc']:.4f} | Val F1: {res['val_macro_f1']:.4f} | Pred Counts: {res['pred_counts']}")

    # 3. PneumoniaMNIST Ablations
    print("\n--- [Ablation: PneumoniaMNIST] ---")
    trials_pneu = [
        ("112x112 + Project + Inverse weights", dict(resolution=112, norm_type="tanh", class_weight_type="inverse")),
        ("112x112 + Replicate 3-ch + ImageNet Norm + Unweighted", dict(resolution=112, norm_type="imagenet", channel_strategy="replicate", class_weight_type="none")),
        ("112x112 + Replicate 3-ch + ImageNet Norm + Sqrt-weighted", dict(resolution=112, norm_type="imagenet", channel_strategy="replicate", class_weight_type="sqrt")),
        ("112x112 + Replicate 3-ch + Linear Probe (Frozen) + lr=1e-3", dict(resolution=112, norm_type="imagenet", channel_strategy="replicate", freeze_backbone=True, class_weight_type="none", lr=1e-3))
    ]
    for name, params in trials_pneu:
        res = run_ablation_trial("pneumoniamnist", **params)
        print(f"  {name}:")
        print(f"    --> Val Bal Acc: {res['val_bal_acc']:.4f} | Val F1: {res['val_macro_f1']:.4f} | Pred Counts: {res['pred_counts']}")

if __name__ == "__main__":
    main()
