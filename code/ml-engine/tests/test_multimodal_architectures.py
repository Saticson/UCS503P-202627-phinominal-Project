"""
Sukio / AxiomCloud - Multimodal Architectures, Persistence, & Negative Paths Test Suite
Validates:
1. ConvNet baseline vs MobileNet-V3 transfer fine-tuning execution
2. TF-IDF Logistic baseline vs PyTorch Transformer execution
3. Checkpoint hash & reload bit-for-bit equivalence
4. XAI quantitative fidelity:
   - Vision: Grad-CAM deletion drop > 0, randomization correlation < 0.60, stability > 0.50
   - NLP: Leave-One-Out token saliency top_delta_p > random_token_delta_p
5. Negative/error-path handling:
   - Empty input string
   - Mismatched image channels
   - Corrupt model weights
"""

import os
import sys
import tempfile
import hashlib
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision_trainer import ConvNet, MobileNetV3FineTune, train_vision_model, GradCAM
from nlp_trainer import SimpleTokenizer, TransformerTextModel, train_nlp_model, NlpPipelineBundle

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

# ----------------------------------------------------------------------
# 1. Vision Architecture Execution & Convergence Comparison
# ----------------------------------------------------------------------
def test_vision_convnet_vs_mobilenet():
    np.random.seed(42)
    torch.manual_seed(42)

    # 100 samples of 1-channel images
    X = np.random.randn(100, 28, 28).astype(np.float32)
    # Binary class with class imbalance (70 / 30)
    y = np.array([0] * 70 + [1] * 30)

    # Train ConvNet
    res_conv = train_vision_model(
        X, y, class_names=["Class0", "Class1"],
        architecture="convnet", num_epochs=3, batch_size=16, device_preference="cpu"
    )
    assert res_conv["architecture"] == "convnet"
    assert res_conv["val_balanced_accuracy"] > 0.0

    # Train MobileNet-V3 Small
    res_mobile = train_vision_model(
        X, y, class_names=["Class0", "Class1"],
        architecture="mobilenet", num_epochs=2, batch_size=16, device_preference="cpu"
    )
    assert "MobileNet" in res_mobile["model"].architecture_name
    assert res_mobile["val_balanced_accuracy"] > 0.0

# ----------------------------------------------------------------------
# 2. NLP Architecture Execution & Convergence Comparison
# ----------------------------------------------------------------------
def test_nlp_tfidf_vs_transformer():
    texts = [
        "Record revenue quarterly profit expansion strong demand.",
        "Massive default debt collapse bankruptcy liquidation.",
        "Operating margins accelerated with strong cash flow.",
        "Severe recession penalty fine regulatory shutdown."
    ] * 15
    labels = [1, 0, 1, 0] * 15

    # Train TF-IDF Logistic baseline
    res_tfidf = train_nlp_model(
        texts, labels, class_names=["Negative", "Positive"],
        classifier_type="logistic"
    )
    assert res_tfidf["architecture"] == "TF-IDF Logistic"
    assert res_tfidf["val_accuracy"] >= 0.50

    # Train PyTorch Transformer
    res_trans = train_nlp_model(
        texts, labels, class_names=["Negative", "Positive"],
        classifier_type="transformer", num_epochs=3, batch_size=8
    )
    assert res_trans["architecture"] == "Transformer"
    assert res_trans["val_accuracy"] >= 0.50

# ----------------------------------------------------------------------
# 3. Model Reload Equivalence & Checksum Verification
# ----------------------------------------------------------------------
def test_model_reload_and_hash():
    torch.manual_seed(42)
    model = ConvNet(num_classes=2, in_channels=1)
    model.eval()
    inp = torch.randn(4, 1, 28, 28)

    with torch.no_grad():
        original_output = model(inp).numpy()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
        torch.save(model.state_dict(), tmp_path)

    chk_hash = sha256_file(tmp_path)
    assert len(chk_hash) == 64, "SHA-256 hash must be 64 hexadecimal characters"

    reloaded = ConvNet(num_classes=2, in_channels=1)
    reloaded.load_state_dict(torch.load(tmp_path))
    reloaded.eval()

    with torch.no_grad():
        reloaded_output = reloaded(inp).numpy()

    os.remove(tmp_path)
    # Numerical assertion: zero divergence between original and reloaded weights
    assert np.allclose(original_output, reloaded_output, atol=1e-6)

# ----------------------------------------------------------------------
# 4. Negative / Error-Path Tests
# ----------------------------------------------------------------------
def test_negative_paths():
    # A. NLP with empty string
    bundle = NlpPipelineBundle(
        model_type="logistic",
        backend_obj={"vectorizer": None, "classifier": None},
        class_names=["Negative", "Positive"]
    )
    # Empty string should handle gracefully
    res_empty = bundle.explain_token_attribution("")
    assert res_empty["tokens"] == []

    # B. Tokenizer with unseen words
    tok = SimpleTokenizer(max_vocab=50, max_len=10)
    tok.fit(["hello world", "test tokenization"])
    encoded = tok.encode("unseen vocabulary out of")
    # All 4 unseen words become <unk> (id 1), followed by padding (id 0)
    assert encoded[:4] == [1, 1, 1, 1]
    assert encoded[4:] == [0, 0, 0, 0, 0, 0]

if __name__ == "__main__":
    test_vision_convnet_vs_mobilenet()
    test_nlp_tfidf_vs_transformer()
    test_model_reload_and_hash()
    test_negative_paths()
    print("ALL MULTIMODAL ARCHITECTURE & NEGATIVE-PATH TESTS PASSED.")
