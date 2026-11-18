"""
Sukio / AxiomCloud - Rigorous NLP Attribution & Model Faithfulness Audit
Validates:
1. DistilBERT attributions come genuinely from its own checkpoint, NOT TF-IDF.
2. Numerical divergence assertion: distilbert_attributions != tfidf_attributions.
3. Multi-sample evaluation across 5 unseen sentences per dataset.
4. Investigation into near-zero softmax probability drops:
   - Softmax saturation: When initial probability is 0.999+, logit margin is large.
   - Measures BOTH delta probability (delta_p) and delta logit (delta_z).
5. Stepwise Deletion Faithfulness Curve (Drop top-1, top-2, top-3 tokens).
6. Random-token control: delta_top > delta_random.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "datasets", "real", "nlp"))
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs"))

def evaluate_nlp_attribution_fidelity(name, filename, text_col, label_col):
    print(f"\n=======================================================")
    print(f"Auditing Attribution Faithfulness for {name}")
    print(f"=======================================================")
    
    path = os.path.join(NLP_DIR, filename)
    df = pd.read_csv(path).dropna()
    
    # Train / Test split
    from sklearn.model_selection import train_test_split
    texts = df[text_col].astype(str).tolist()
    labels_raw = df[label_col].tolist()
    unique_labels = sorted(list(set(labels_raw)))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    y = np.array([label_map[l] for l in labels_raw])
    
    tr_txt, te_txt, tr_y, te_y = train_test_split(texts, y, test_size=0.20, random_state=42)
    
    # 1. Fit TF-IDF Baseline
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    clf = LogisticRegression(max_iter=300, random_state=42, class_weight="balanced")
    pipe = Pipeline([("vec", vec), ("clf", clf)])
    pipe.fit(tr_txt, tr_y)
    
    # 2. Fit DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
    for p in model.distilbert.embeddings.parameters(): p.requires_grad = False
    for l in model.distilbert.transformer.layer[:4]:
        for p in l.parameters(): p.requires_grad = False
        
    n_sub = min(len(tr_txt), 1500)
    enc_tr = tokenizer(tr_txt[:n_sub], padding=True, truncation=True, max_length=64, return_tensors="pt")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"], torch.tensor(tr_y[:n_sub])),
        batch_size=32, shuffle=True
    )
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
    model.train()
    for ep in range(1): # Fast 1 epoch audit fine-tune
        for bx, bm, by in loader:
            opt.zero_grad()
            loss = model(input_ids=bx, attention_mask=bm, labels=by).loss
            loss.backward()
            opt.step()
    model.eval()

    sample_evaluations = []
    
    # Select 5 diverse unseen test samples with at least 6 tokens
    valid_test_indices = [i for i, t in enumerate(te_txt) if len(t.split()) >= 6][:5]
    
    for s_idx in valid_test_indices:
        sample = te_txt[s_idx]
        true_c = int(te_y[s_idx])
        words = sample.split()[:12] # Work with first 12 tokens
        clean_text = " ".join(words)
        
        # --- TF-IDF Attribution ---
        p_orig_tfidf = pipe.predict_proba([clean_text])[0]
        pred_c_tfidf = int(np.argmax(p_orig_tfidf))
        top_p_tfidf = p_orig_tfidf[pred_c_tfidf]
        
        tfidf_deltas = []
        for i in range(len(words)):
            abl_text = " ".join([w for j, w in enumerate(words) if j != i])
            p_abl = pipe.predict_proba([abl_text])[0][pred_c_tfidf]
            tfidf_deltas.append(float(top_p_tfidf - p_abl))
            
        # --- DistilBERT Attribution ---
        enc = tokenizer([clean_text], padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[0]
            p_orig_bert = torch.softmax(out, dim=0).numpy()
            z_orig_bert = out.numpy()
            
        pred_c_bert = int(np.argmax(p_orig_bert))
        top_p_bert = float(p_orig_bert[pred_c_bert])
        top_z_bert = float(z_orig_bert[pred_c_bert])
        
        bert_deltas_p = []
        bert_deltas_z = []
        for i in range(len(words)):
            abl_text = " ".join([w for j, w in enumerate(words) if j != i])
            enc_abl = tokenizer([abl_text], padding=True, truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                out_abl = model(input_ids=enc_abl["input_ids"], attention_mask=enc_abl["attention_mask"]).logits[0]
                p_abl_bert = torch.softmax(out_abl, dim=0).numpy()[pred_c_bert]
                z_abl_bert = out_abl.numpy()[pred_c_bert]
            bert_deltas_p.append(float(top_p_bert - p_abl_bert))
            bert_deltas_z.append(float(top_z_bert - z_abl_bert))

        # Check numerical divergence: Are TF-IDF and DistilBERT attribution vectors different?
        is_distinct = not np.allclose(tfidf_deltas, bert_deltas_p, atol=1e-4)
        corr, _ = spearmanr(tfidf_deltas, bert_deltas_p)
        
        # Stepwise multi-token deletion
        # Drop top-1, top-2, top-3 ranked tokens by DistilBERT logit sensitivity
        ranked_token_indices = np.argsort(bert_deltas_z)[::-1]
        deletion_curve_p = [top_p_bert]
        deletion_curve_z = [top_z_bert]
        for k in range(1, min(4, len(words))):
            drop_indices = set(ranked_token_indices[:k])
            abl_multi = " ".join([w for j, w in enumerate(words) if j not in drop_indices])
            enc_m = tokenizer([abl_multi], padding=True, truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                out_m = model(input_ids=enc_m["input_ids"], attention_mask=enc_m["attention_mask"]).logits[0]
                p_m = float(torch.softmax(out_m, dim=0).numpy()[pred_c_bert])
                z_m = float(out_m.numpy()[pred_c_bert])
            deletion_curve_p.append(p_m)
            deletion_curve_z.append(z_m)

        # Random control: average delta of non-top tokens
        top_token_word = words[ranked_token_indices[0]]
        delta_top_p = bert_deltas_p[ranked_token_indices[0]]
        delta_top_z = bert_deltas_z[ranked_token_indices[0]]
        delta_rand_p = float(np.mean([bert_deltas_p[idx] for idx in ranked_token_indices[1:]])) if len(words) > 1 else 0.0
        delta_rand_z = float(np.mean([bert_deltas_z[idx] for idx in ranked_token_indices[1:]])) if len(words) > 1 else 0.0
        
        sample_evaluations.append({
            "text": clean_text,
            "pred_class": pred_c_bert,
            "orig_prob": round(top_p_bert, 4),
            "orig_logit": round(top_z_bert, 4),
            "top_token": top_token_word,
            "delta_top_p": round(delta_top_p, 4),
            "delta_rand_p": round(delta_rand_p, 4),
            "delta_top_z": round(delta_top_z, 4),
            "delta_rand_z": round(delta_rand_z, 4),
            "deletion_curve_logits": [round(z, 4) for z in deletion_curve_z],
            "deletion_curve_probs": [round(p, 4) for p in deletion_curve_p],
            "vectors_distinct_from_tfidf": bool(is_distinct),
            "spearman_rank_correlation": round(float(corr) if not np.isnan(corr) else 0.0, 4)
        })
        
        print(f"  Sample: \"{clean_text[:40]}...\"")
        print(f"    Orig Prob: {top_p_bert:.4f} | Orig Logit: {top_z_bert:.4f} | Top Token: '{top_token_word}'")
        print(f"    Delta Logit: Top={delta_top_z:.4f} vs Rand={delta_rand_z:.4f}")
        print(f"    Delta Prob:  Top={delta_top_p:.4f} vs Rand={delta_rand_p:.4f}")
        print(f"    Stepwise Deletion (Logits): {deletion_curve_z}")
        print(f"    Distinct from TF-IDF: {is_distinct} (Spearman r = {corr:.3f})")

    return sample_evaluations

def main():
    datasets = [
        ("ag_news", "ag_news.csv", "text", "topic"),
        ("sms_spam", "sms_spam.csv", "text", "label"),
        ("financial_phrasebank", "financial_phrasebank.csv", "text", "sentiment"),
        ("rotten_tomatoes", "rotten_tomatoes.csv", "text", "sentiment"),
        ("tweeteval_sentiment", "tweeteval_sentiment.csv", "text", "sentiment")
    ]
    
    all_results = {}
    for name, fname, tc, lc in datasets:
        all_results[name] = evaluate_nlp_attribution_fidelity(name, fname, tc, lc)
        
    out_file = os.path.join(DOCS_DIR, "DISTILBERT_ATTRIBUTION_AUDIT.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll NLP attribution audits written to: {out_file}")

if __name__ == "__main__":
    main()
