"""
Sukio / AxiomCloud - Unified Multimodal ML Engine
Executes genuine end-to-end Training, Inference, and XAI across Tabular, Vision, and NLP.
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def handle_train_vision(args):
    """Expects JSON payload with image_data (paths or base64) and labels."""
    from vision_trainer import train_vision_model, save_vision_artifact
    with open(args.input, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    images = payload["images"]
    labels = payload["labels"]
    class_names = payload.get("class_names", [f"Class_{i}" for i in range(len(set(labels)))])
    architecture = payload.get("architecture", "convnet")
    epochs = payload.get("epochs", 10)
    batch_size = payload.get("batch_size", 32)
    lr = payload.get("lr", 0.001)

    res = train_vision_model(
        images, labels, class_names,
        architecture=architecture,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )
    model = res["model"]
    artifact_path = args.output or "./storage/models/vision_model.pt"
    save_vision_artifact(model, class_names, artifact_path, conformal_quantiles=res["conformal_quantiles"], architecture=architecture)

    output = {
        "status": "success",
        "modality": "COMPUTER_VISION",
        "architecture": res["architecture"],
        "hardware_name": res["hardware_name"],
        "device": res["device"],
        "val_accuracy": res["val_accuracy"],
        "val_balanced_accuracy": res["val_balanced_accuracy"],
        "val_macro_f1": res["val_macro_f1"],
        "val_loss": res["val_loss"],
        "confusion_matrix": res["confusion_matrix"],
        "conformal_quantiles": res["conformal_quantiles"],
        "xai_fidelity": res["xai_fidelity"],
        "train_time_sec": res["train_time_sec"],
        "artifact_path": os.path.abspath(artifact_path),
        "class_names": class_names
    }
    print(json.dumps(output))

def handle_predict_vision(args):
    """Loads vision model, runs inference on image, computes Grad-CAM heatmap & fidelity."""
    import torch
    from PIL import Image
    from vision_trainer import load_vision_artifact, GradCAM

    model, class_names, conformal_quantiles = load_vision_artifact(args.model_path)
    img = Image.open(args.image_path).convert('RGB').resize((28, 28))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor - 0.5) / 0.5

    gradcam = GradCAM(model)
    heatmap, pred_idx, probs = gradcam.generate_heatmap(img_tensor)
    fidelity = gradcam.evaluate_explanation_faithfulness(img_tensor, pred_idx)

    # Class-Conditional Conformal Set (target 90%)
    conf_set = []
    for c in range(len(class_names)):
        q = conformal_quantiles.get(str(c), conformal_quantiles.get(c, 0.90))
        if probs[c] >= (1.0 - q):
            conf_set.append(class_names[c])
    if not conf_set:
        conf_set = [class_names[pred_idx]]

    output = {
        "status": "success",
        "modality": "COMPUTER_VISION",
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "conformal_prediction_set": conf_set,
        "gradcam_heatmap": heatmap.tolist(),
        "heatmap_shape": list(heatmap.shape),
        "xai_fidelity": fidelity
    }
    print(json.dumps(output))

def handle_train_nlp(args):
    """Expects JSON payload with texts and labels."""
    from nlp_trainer import train_nlp_model, save_nlp_artifact
    with open(args.input, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    texts = payload["texts"]
    labels = payload["labels"]
    class_names = payload.get("class_names", [f"Class_{i}" for i in range(len(set(labels)))])
    classifier_type = payload.get("classifier_type", "logistic")
    epochs = payload.get("epochs", 8)
    batch_size = payload.get("batch_size", 32)
    lr = payload.get("lr", 0.001)

    res = train_nlp_model(
        texts, labels, class_names,
        classifier_type=classifier_type,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )
    bundle = res["bundle"]
    artifact_path = args.output or "./storage/models/nlp_model.joblib"
    save_nlp_artifact(bundle, artifact_path)

    output = {
        "status": "success",
        "modality": "NLP",
        "architecture": res["architecture"],
        "val_accuracy": res["val_accuracy"],
        "val_macro_f1": res["val_macro_f1"],
        "val_balanced_accuracy": res["val_balanced_accuracy"],
        "confusion_matrix": res["confusion_matrix"],
        "conformal_quantiles": res["conformal_quantiles"],
        "train_time_sec": res["train_time_sec"],
        "artifact_path": os.path.abspath(artifact_path),
        "class_names": class_names
    }
    print(json.dumps(output))

def handle_predict_nlp(args):
    """Loads NLP model, runs inference on text, computes leave-one-out token attribution & fidelity."""
    from nlp_trainer import load_nlp_artifact
    bundle = load_nlp_artifact(args.model_path)
    pred_res = bundle.predict(args.text)
    xai_res = bundle.explain_token_attribution(args.text)

    # Class-Conditional Conformal Set
    conf_set = []
    q_dict = getattr(bundle, 'conformal_quantiles', {})
    probs = pred_res["probabilities"]
    for i, cname in enumerate(bundle.class_names):
        q = q_dict.get(str(i), q_dict.get(i, 0.90))
        if probs.get(cname, 0.0) >= (1.0 - q):
            conf_set.append(cname)
    if not conf_set:
        conf_set = [pred_res["prediction"]]

    output = {
        "status": "success",
        "modality": "NLP",
        "architecture": pred_res.get("architecture", "NLP Pipeline"),
        "prediction": pred_res["prediction"],
        "confidence": pred_res["confidence"],
        "probabilities": pred_res["probabilities"],
        "conformal_prediction_set": conf_set,
        "token_attribution": xai_res["tokens"],
        "xai_fidelity": xai_res.get("fidelity", {})
    }
    print(json.dumps(output))

def main():
    parser = argparse.ArgumentParser(description="Sukio Multimodal ML Engine")
    subparsers = parser.add_subparsers(dest="command")

    p_vtrain = subparsers.add_parser("train-vision")
    p_vtrain.add_argument("--input", required=True, help="Path to input JSON spec")
    p_vtrain.add_argument("--output", help="Path to output .pt artifact")

    p_vpred = subparsers.add_parser("predict-vision")
    p_vpred.add_argument("--model-path", required=True, help="Path to .pt artifact")
    p_vpred.add_argument("--image-path", required=True, help="Path to input image file")

    p_ntrain = subparsers.add_parser("train-nlp")
    p_ntrain.add_argument("--input", required=True, help="Path to input JSON spec")
    p_ntrain.add_argument("--output", help="Path to output .joblib artifact")

    p_npred = subparsers.add_parser("predict-nlp")
    p_npred.add_argument("--model-path", required=True, help="Path to .joblib artifact")
    p_npred.add_argument("--text", required=True, help="Input text string")

    args = parser.parse_args()
    if args.command == "train-vision":
        handle_train_vision(args)
    elif args.command == "predict-vision":
        handle_predict_vision(args)
    elif args.command == "train-nlp":
        handle_train_nlp(args)
    elif args.command == "predict-nlp":
        handle_predict_nlp(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
