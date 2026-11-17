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
    epochs = payload.get("epochs", 5)

    res = train_vision_model(images, labels, class_names, num_epochs=epochs)
    model = res["model"]
    artifact_path = args.output or "./storage/models/vision_model.pt"
    save_vision_artifact(model, class_names, artifact_path)

    output = {
        "status": "success",
        "modality": "COMPUTER_VISION",
        "val_accuracy": res["val_accuracy"],
        "val_loss": res["val_loss"],
        "train_time_sec": res["train_time_sec"],
        "artifact_path": os.path.abspath(artifact_path),
        "class_names": class_names
    }
    print(json.dumps(output))

def handle_predict_vision(args):
    """Loads vision model, runs inference on image, computes Grad-CAM heatmap."""
    import torch
    from PIL import Image
    from vision_trainer import load_vision_artifact, GradCAM

    model, class_names = load_vision_artifact(args.model_path)
    img = Image.open(args.image_path).convert('RGB').resize((28, 28))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor - 0.5) / 0.5

    gradcam = GradCAM(model, model.features[-1])
    heatmap, pred_idx, probs = gradcam.generate_heatmap(img_tensor)

    output = {
        "status": "success",
        "modality": "COMPUTER_VISION",
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "gradcam_heatmap": heatmap.tolist(),
        "heatmap_shape": list(heatmap.shape)
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

    res = train_nlp_model(texts, labels, class_names, classifier_type=classifier_type)
    bundle = res["bundle"]
    artifact_path = args.output or "./storage/models/nlp_model.joblib"
    save_nlp_artifact(bundle, artifact_path)

    output = {
        "status": "success",
        "modality": "NLP",
        "val_accuracy": res["val_accuracy"],
        "val_macro_f1": res["val_macro_f1"],
        "val_balanced_accuracy": res["val_balanced_accuracy"],
        "train_time_sec": res["train_time_sec"],
        "artifact_path": os.path.abspath(artifact_path),
        "class_names": class_names
    }
    print(json.dumps(output))

def handle_predict_nlp(args):
    """Loads NLP model, runs inference on text, computes leave-one-out token attribution."""
    from nlp_trainer import load_nlp_artifact
    bundle = load_nlp_artifact(args.model_path)
    pred_res = bundle.predict(args.text)
    xai_res = bundle.explain_token_attribution(args.text)

    output = {
        "status": "success",
        "modality": "NLP",
        "prediction": pred_res["prediction"],
        "confidence": pred_res["confidence"],
        "probabilities": pred_res["probabilities"],
        "token_attribution": xai_res["tokens"]
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
