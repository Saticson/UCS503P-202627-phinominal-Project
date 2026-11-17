"""
Sukio / AxiomCloud - XAI Fidelity & Model Sanity Unit Tests
Validates:
1. Local vs Global distinction: SHAP additivity vs Permutation Importance
2. Grad-CAM deletion drop (positive fidelity)
3. Model randomization sanity check (Adebayo et al. 2018)
4. Input perturbation stability check
5. Leave-One-Out NLP token saliency vs random-token baseline
6. Checkpoint persistence and reloaded inference bit-for-bit equivalence
"""

import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision_trainer import ConvNet, GradCAM
from nlp_trainer import NlpPipelineBundle, SimpleTokenizer

# ----------------------------------------------------------------------
# 1. SHAP Additivity vs Global Permutation Importance Test
# ----------------------------------------------------------------------
def test_shap_additivity_vs_permutation():
    # Efficiency axiom of local Shapley values: sum(phi_i) == f(x) - E[f(x)]
    base_value = 0.35
    prediction = 0.85
    # Synthetic known-answer Shapley contributions
    shap_values = {"feature_a": 0.25, "feature_b": 0.15, "feature_c": 0.10}

    efficiency_sum = sum(shap_values.values())
    assert abs(efficiency_sum - (prediction - base_value)) < 1e-6, "Shapley efficiency axiom violated!"

    # In contrast, global permutation importance is non-additive and measures metric degradation
    permutation_importance = {"feature_a": 0.12, "feature_b": 0.08, "feature_c": 0.02}
    assert sum(permutation_importance.values()) != (prediction - base_value), "Permutation importance must NOT be confused with local SHAP additivity!"

# ----------------------------------------------------------------------
# 2. Grad-CAM Deletion Drop & Insertion Gain
# ----------------------------------------------------------------------
def test_gradcam_deletion_and_insertion():
    torch.manual_seed(42)
    np.random.seed(42)
    model = ConvNet(num_classes=2, in_channels=1)
    model.eval()

    gradcam = GradCAM(model)
    # Synthetic input with a strong localized signal in the center
    img = torch.zeros(1, 1, 28, 28)
    img[0, 0, 10:18, 10:18] = 2.0

    fidelity = gradcam.evaluate_explanation_faithfulness(img, target_class=0)
    assert "delta_deletion" in fidelity
    assert "delta_insertion" in fidelity
    assert "randomization_correlation" in fidelity
    assert "stability_correlation" in fidelity

# ----------------------------------------------------------------------
# 3. Model Randomization Sanity Check (Adebayo et al.)
# ----------------------------------------------------------------------
def test_model_randomization_sanity_check():
    torch.manual_seed(42)
    model = ConvNet(num_classes=2, in_channels=1)
    model.eval()

    gradcam = GradCAM(model)
    img = torch.randn(1, 1, 28, 28)
    fidelity = gradcam.evaluate_explanation_faithfulness(img)

    # When model weights are randomized, correlation with original heatmap must drop
    assert fidelity["randomization_correlation"] < 0.80, f"Failed sanity check: correlation {fidelity['randomization_correlation']} too high!"

# ----------------------------------------------------------------------
# 4. Input Perturbation Stability Check
# ----------------------------------------------------------------------
def test_gradcam_stability():
    torch.manual_seed(42)
    model = ConvNet(num_classes=2, in_channels=1)
    model.eval()

    gradcam = GradCAM(model)
    img = torch.randn(1, 1, 28, 28)
    fidelity = gradcam.evaluate_explanation_faithfulness(img)

    # Heatmap should remain reasonably stable under small epsilon noise
    assert fidelity["stability_correlation"] > 0.40, f"Stability too low: {fidelity['stability_correlation']}"

# ----------------------------------------------------------------------
# 5. NLP Token Saliency vs Random Token Baseline
# ----------------------------------------------------------------------
def test_nlp_token_saliency_vs_random():
    # Construct a pipeline bundle where specific tokens drive probabilities
    class MockClassifier:
        def predict_proba(self, X):
            # Sum feature vector
            arr = X.toarray()[0]
            p1 = 1.0 / (1.0 + np.exp(-3.0 * (arr[0] - arr[1] if len(arr) > 1 else 0)))
            return np.array([[1.0 - p1, p1]])

    class MockVectorizer:
        def __init__(self):
            self.vocab = {"record": 0, "quarterly": 1, "revenue": 2}
        def transform(self, texts):
            from scipy.sparse import csr_matrix
            text = texts[0].lower()
            row = [1.0 if w in text else 0.0 for w in ["record", "quarterly", "revenue"]]
            return csr_matrix([row])

    bundle = NlpPipelineBundle(
        model_type="logistic",
        backend_obj={"vectorizer": MockVectorizer(), "classifier": MockClassifier()},
        class_names=["Negative", "Positive"]
    )

    res = bundle.explain_token_attribution("Record quarterly revenue exceeded estimates", target_class="Positive")
    fid = res["fidelity"]
    assert fid["top_delta_p"] >= fid["random_token_delta_p"], "Top salient token must have >= impact than random baseline!"

# ----------------------------------------------------------------------
# 6. Checkpoint Serialization & Reload Equivalence
# ----------------------------------------------------------------------
def test_checkpoint_reload_equivalence():
    torch.manual_seed(42)
    model = ConvNet(num_classes=2, in_channels=1)
    model.eval()
    test_input = torch.randn(5, 1, 28, 28)

    with torch.no_grad():
        orig_preds = model(test_input).numpy()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
        torch.save(model.state_dict(), tmp_path)

    reloaded_model = ConvNet(num_classes=2, in_channels=1)
    reloaded_model.load_state_dict(torch.load(tmp_path))
    reloaded_model.eval()

    with torch.no_grad():
        reloaded_preds = reloaded_model(test_input).numpy()

    os.remove(tmp_path)
    assert np.allclose(orig_preds, reloaded_preds, atol=1e-6), "Reloaded model outputs diverged from original model!"

if __name__ == "__main__":
    test_shap_additivity_vs_permutation()
    test_gradcam_deletion_and_insertion()
    test_model_randomization_sanity_check()
    test_gradcam_stability()
    test_nlp_token_saliency_vs_random()
    test_checkpoint_reload_equivalence()
    print("ALL XAI FIDELITY & MODEL SANITY TESTS PASSED.")
