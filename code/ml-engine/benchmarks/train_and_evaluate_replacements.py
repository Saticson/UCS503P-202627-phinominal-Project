import os
import sys
import json
import hashlib
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision_trainer import ConvNet, MobileNetV3FineTune
from nlp_trainer import TransformerTextModel, SimpleTokenizer, NlpPipelineBundle, clean_text

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_TAB = os.path.join(REPO_ROOT, "code", "storage", "datasets", "real", "tabular")
DATA_VIS = os.path.join(REPO_ROOT, "code", "storage", "datasets", "real", "vision")
DATA_NLP = os.path.join(REPO_ROOT, "code", "storage", "datasets", "real", "nlp")
MODEL_DIR = os.path.join(REPO_ROOT, "code", "storage", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def train_and_eval_pathmnist():
    print("=" * 70)
    print("1. VISION REPLACEMENT: PathMNIST (9 classes, Colorectal Histology)")
    print("=" * 70)
    path = os.path.join(DATA_VIS, "pathmnist.npz")
    d_sha = sha256_file(path)
    data = np.load(path)
    
    x_tr = data["train_images"]
    y_tr = data["train_labels"].squeeze()
    x_te = data["test_images"]
    y_te = data["test_labels"].squeeze()
    
    num_classes = 9
    in_channels = 3
    counts_true = np.bincount(y_te, minlength=num_classes)
    
    print(f"PathMNIST: Train N={len(y_tr)}, Test N={len(y_te)}, Classes={num_classes}")
    print(f"Ground Truth Test Class Counts: {counts_true.tolist()}")
    
    # Fast tensor conversion
    t_tr_x = torch.tensor(np.transpose(x_tr[:10000].astype(np.float32) / 255.0, (0, 3, 1, 2)) - 0.5) / 0.5
    t_tr_y = torch.tensor(y_tr[:10000], dtype=torch.long)
    t_te_x = torch.tensor(np.transpose(x_te.astype(np.float32) / 255.0, (0, 3, 1, 2)) - 0.5) / 0.5
    t_te_y = torch.tensor(y_te, dtype=torch.long)
    
    loader_tr = DataLoader(TensorDataset(t_tr_x, t_tr_y), batch_size=64, shuffle=True)
    loader_te = DataLoader(TensorDataset(t_te_x, t_te_y), batch_size=128, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    # A. ConvNet Baseline
    print("--- Training ConvNet Baseline ---")
    convnet = ConvNet(num_classes=num_classes, in_channels=in_channels).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(convnet.parameters(), lr=0.002, weight_decay=1e-4)
    convnet.train()
    for ep in range(3):
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(convnet(bx), by)
            loss.backward()
            opt.step()
            
    convnet.eval()
    convnet.cpu()
    with torch.no_grad():
        logits = convnet(t_te_x)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
    cm = confusion_matrix(y_te, preds, labels=list(range(num_classes)))
    bal_acc = float(balanced_accuracy_score(y_te, preds))
    macro_f1 = float(f1_score(y_te, preds, average='macro', zero_division=0))
    auroc = float(roc_auc_score(y_te, probs, multi_class='ovr'))
    
    chk_conv = os.path.join(MODEL_DIR, "pathmnist_convnet_baseline.pth")
    torch.save(convnet.state_dict(), chk_conv)
    print(f"ConvNet -> Bal Acc: {bal_acc:.4f}, Macro-F1: {macro_f1:.4f}, AUROC: {auroc:.4f}")
    
    results["ConvNet Baseline"] = {
        "checkpoint": chk_conv,
        "checkpoint_sha": sha256_file(chk_conv),
        "confusion_matrix": cm.tolist(),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "auroc": round(auroc, 4),
        "row_sums_true": cm.sum(axis=1).tolist(),
        "col_sums_pred": cm.sum(axis=0).tolist()
    }
    
    # B. MobileNet-V3 Small
    print("--- Training MobileNet-V3 Small ---")
    mobilenet = MobileNetV3FineTune(num_classes=num_classes, in_channels=in_channels, pretrained=False).to(device)
    opt_m = optim.AdamW(mobilenet.parameters(), lr=0.001, weight_decay=1e-4)
    mobilenet.train()
    for ep in range(2):
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            opt_m.zero_grad()
            loss = crit(mobilenet(bx), by)
            loss.backward()
            opt_m.step()
            
    mobilenet.eval()
    mobilenet.cpu()
    with torch.no_grad():
        logits_m = mobilenet(t_te_x)
        probs_m = torch.softmax(logits_m, dim=1).numpy()
        preds_m = np.argmax(probs_m, axis=1)
        
    cm_m = confusion_matrix(y_te, preds_m, labels=list(range(num_classes)))
    bal_acc_m = float(balanced_accuracy_score(y_te, preds_m))
    macro_f1_m = float(f1_score(y_te, preds_m, average='macro', zero_division=0))
    auroc_m = float(roc_auc_score(y_te, probs_m, multi_class='ovr'))
    
    chk_mob = os.path.join(MODEL_DIR, "pathmnist_pretrained_mobilenet-v3_small.pth")
    torch.save(mobilenet.state_dict(), chk_mob)
    print(f"MobileNet-V3 -> Bal Acc: {bal_acc_m:.4f}, Macro-F1: {macro_f1_m:.4f}, AUROC: {auroc_m:.4f}")
    
    results["MobileNet-V3 Small"] = {
        "checkpoint": chk_mob,
        "checkpoint_sha": sha256_file(chk_mob),
        "confusion_matrix": cm_m.tolist(),
        "balanced_accuracy": round(bal_acc_m, 4),
        "macro_f1": round(macro_f1_m, 4),
        "auroc": round(auroc_m, 4),
        "row_sums_true": cm_m.sum(axis=1).tolist(),
        "col_sums_pred": cm_m.sum(axis=0).tolist()
    }
    
    return {
        "dataset": "pathmnist",
        "dataset_sha": d_sha,
        "test_sample_count": len(y_te),
        "classes": num_classes,
        "true_distribution": counts_true.tolist(),
        "models": results
    }

def train_and_eval_california_housing():
    print("\n" + "=" * 70)
    print("2. TABULAR REPLACEMENT: California Housing Regression (20,433 rows)")
    print("=" * 70)
    p_tr = os.path.join(DATA_TAB, "california_housing_train.csv")
    p_te = os.path.join(DATA_TAB, "california_housing_test.csv")
    df_tr = pd.read_csv(p_tr)
    df_te = pd.read_csv(p_te)
    
    target = "median_house_value"
    X_tr = df_tr.drop(columns=[target])
    y_tr = df_tr[target].values
    X_te = df_te.drop(columns=[target])
    y_te = df_te[target].values
    
    cat_cols = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_tr.select_dtypes(include=['int64', 'float64']).columns.tolist()
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    pipe = Pipeline([('pre', pre), ('reg', reg)])
    print(f"Fitting Random Forest Regressor on {len(X_tr)} train rows...")
    pipe.fit(X_tr, y_tr)
    
    preds = pipe.predict(X_te)
    r2 = float(r2_score(y_te, preds))
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    mae = float(mean_absolute_error(y_te, preds))
    
    chk_path = os.path.join(MODEL_DIR, "california_housing_rf.joblib")
    joblib.dump(pipe, chk_path)
    print(f"California Housing RF -> R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    return {
        "dataset": "california_housing",
        "dataset_sha_train": sha256_file(p_tr),
        "dataset_sha_test": sha256_file(p_te),
        "total_rows": len(df_tr) + len(df_te),
        "test_rows": len(df_te),
        "task": "Continuous Regression",
        "models": {
            "Random Forest Regressor": {
                "checkpoint": chk_path,
                "checkpoint_sha": sha256_file(chk_path),
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "test_rows": len(y_te)
            }
        }
    }

def train_and_eval_sst2():
    print("\n" + "=" * 70)
    print("3. NLP REPLACEMENT: SST-2 Sentiment Classification (8,741 sentences)")
    print("=" * 70)
    p_tr = os.path.join(DATA_NLP, "sst2_train.csv")
    p_te = os.path.join(DATA_NLP, "sst2_test.csv")
    df_tr = pd.read_csv(p_tr).dropna()
    df_te = pd.read_csv(p_te).dropna()
    
    X_tr = [clean_text(t) for t in df_tr["text"].tolist()]
    y_tr = df_tr["label"].values
    X_te = [clean_text(t) for t in df_te["text"].tolist()]
    y_te = df_te["label"].values
    
    counts_true = np.bincount(y_te, minlength=2)
    print(f"SST-2: Train N={len(X_tr)}, Test N={len(X_te)}")
    print(f"Ground Truth Test Class Counts: {counts_true.tolist()}")
    
    # A. TF-IDF Baseline
    print("--- Training TF-IDF Baseline ---")
    vec = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), sublinear_tf=True)
    clf = LogisticRegression(max_iter=300, random_state=42, class_weight='balanced')
    pipe = Pipeline([('vec', vec), ('clf', clf)])
    pipe.fit(X_tr, y_tr)
    
    preds_tfidf = pipe.predict(X_te)
    probs_tfidf = pipe.predict_proba(X_te)
    
    cm_tfidf = confusion_matrix(y_te, preds_tfidf)
    bal_acc_t = float(balanced_accuracy_score(y_te, preds_tfidf))
    macro_f1_t = float(f1_score(y_te, preds_tfidf, average='macro'))
    auroc_t = float(roc_auc_score(y_te, probs_tfidf[:, 1]))
    
    chk_tfidf = os.path.join(MODEL_DIR, "sst2_tfidf_logistic.joblib")
    bundle_tfidf = NlpPipelineBundle(
        model_type="logistic",
        backend_obj={"vectorizer": vec, "classifier": clf},
        class_names=["Negative", "Positive"]
    )
    joblib.dump(bundle_tfidf, chk_tfidf)
    print(f"TF-IDF -> Bal Acc: {bal_acc_t:.4f}, Macro-F1: {macro_f1_t:.4f}, AUROC: {auroc_t:.4f}")
    
    # B. PyTorch Transformer
    print("--- Training PyTorch Transformer ---")
    tokenizer = SimpleTokenizer(max_vocab=8000, max_len=64)
    tokenizer.fit(X_tr)
    tr_tokens = [tokenizer.encode(t) for t in X_tr]
    te_tokens = [tokenizer.encode(t) for t in X_te]
    
    loader_tr = DataLoader(TensorDataset(torch.tensor(tr_tokens), torch.tensor(y_tr)), batch_size=32, shuffle=True)
    loader_te = DataLoader(TensorDataset(torch.tensor(te_tokens), torch.tensor(y_te)), batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tf = TransformerTextModel(len(tokenizer.vocab), num_classes=2).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model_tf.parameters(), lr=0.001, weight_decay=1e-4)
    
    model_tf.train()
    for ep in range(3):
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model_tf(bx), by)
            loss.backward()
            opt.step()
            
    model_tf.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for bx, by in loader_te:
            bx = bx.to(device)
            p = torch.softmax(model_tf(bx), dim=1).cpu().numpy()
            all_preds.extend(np.argmax(p, axis=1).tolist())
            all_probs.extend(p.tolist())
            
    preds_tf = np.array(all_preds)
    probs_tf = np.array(all_probs)
    
    cm_tf = confusion_matrix(y_te, preds_tf)
    bal_acc_tf = float(balanced_accuracy_score(y_te, preds_tf))
    macro_f1_tf = float(f1_score(y_te, preds_tf, average='macro'))
    auroc_tf = float(roc_auc_score(y_te, probs_tf[:, 1]))
    
    chk_tf = os.path.join(MODEL_DIR, "sst2_transformer.joblib")
    bundle_tf = NlpPipelineBundle(
        model_type="transformer",
        backend_obj=model_tf.cpu(),
        class_names=["Negative", "Positive"],
        tokenizer=tokenizer
    )
    joblib.dump(bundle_tf, chk_tf)
    print(f"Transformer -> Bal Acc: {bal_acc_tf:.4f}, Macro-F1: {macro_f1_tf:.4f}, AUROC: {auroc_tf:.4f}")
    
    return {
        "dataset": "sst2",
        "dataset_sha_train": sha256_file(p_tr),
        "dataset_sha_test": sha256_file(p_te),
        "train_rows": len(X_tr),
        "test_rows": len(X_te),
        "true_distribution": counts_true.tolist(),
        "models": {
            "TF-IDF Baseline": {
                "checkpoint": chk_tfidf,
                "checkpoint_sha": sha256_file(chk_tfidf),
                "confusion_matrix": cm_tfidf.tolist(),
                "balanced_accuracy": round(bal_acc_t, 4),
                "macro_f1": round(macro_f1_t, 4),
                "auroc": round(auroc_t, 4),
                "row_sums_true": cm_tfidf.sum(axis=1).tolist(),
                "col_sums_pred": cm_tfidf.sum(axis=0).tolist()
            },
            "PyTorch Transformer": {
                "checkpoint": chk_tf,
                "checkpoint_sha": sha256_file(chk_tf),
                "confusion_matrix": cm_tf.tolist(),
                "balanced_accuracy": round(bal_acc_tf, 4),
                "macro_f1": round(macro_f1_tf, 4),
                "auroc": round(auroc_tf, 4),
                "row_sums_true": cm_tf.sum(axis=1).tolist(),
                "col_sums_pred": cm_tf.sum(axis=0).tolist()
            }
        }
    }

if __name__ == "__main__":
    t0 = time.time()
    res_path = train_and_eval_pathmnist()
    res_house = train_and_eval_california_housing()
    res_sst = train_and_eval_sst2()
    
    summary = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "replacements": {
            "vision": res_path,
            "tabular": res_house,
            "nlp": res_sst
        },
        "total_time_sec": round(time.time() - t0, 2)
    }
    
    out_path = os.path.join(REPO_ROOT, "docs", "REPLACEMENT_BENCHMARKS_PROVENANCE.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved replacement benchmark provenance to: {out_path}")
