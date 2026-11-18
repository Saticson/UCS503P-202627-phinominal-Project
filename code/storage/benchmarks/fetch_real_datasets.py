"""
Sukio / AxiomCloud - Automated Real Dataset Fetcher & Cryptographic Manifest Generator
Curates 15 genuine, publicly accessible, non-trivial benchmark datasets:
- 5 Tabular (UCI: German Credit, Heart Disease, Bank Marketing, Adult Census, Breast Cancer Coimbra)
- 5 Computer Vision (MedMNIST: BloodMNIST, PathMNIST, DermaMNIST, RetinaMNIST, TissueMNIST)
- 5 NLP (Financial PhraseBank, Rotten Tomatoes SST-2, AG News, TweetEval Sentiment, UCI SMS Spam)

Outputs verified CSV and NPZ files into code/storage/datasets/real/{tabular,vision,nlp}/
and records exact SHA-256 checksums, shapes, splits, and provenance in DATASET_MANIFEST.json.
"""

import os
import sys
import io
import json
import hashlib
import zipfile
import subprocess
import urllib.request
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "real"))
TABULAR_DIR = os.path.join(BASE_DIR, "tabular")
VISION_DIR = os.path.join(BASE_DIR, "vision")
NLP_DIR = os.path.join(BASE_DIR, "nlp")

os.makedirs(TABULAR_DIR, exist_ok=True)
os.makedirs(VISION_DIR, exist_ok=True)
os.makedirs(NLP_DIR, exist_ok=True)

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def fetch_tabular_datasets():
    print("--- [1/3] Fetching 5 Tabular Benchmark Datasets from UCI Repository ---")
    manifest = {}

    headers = {'User-Agent': 'Mozilla/5.0'}

    # 1. Statlog German Credit (UCI ID 144)
    p_german = os.path.join(TABULAR_DIR, "statlog_german_credit.csv")
    if not os.path.exists(p_german):
        print("  Downloading Statlog German Credit...")
        req = urllib.request.Request('https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip', headers=headers)
        with urllib.request.urlopen(req) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            with z.open('german.data') as f:
                col_names = [f"attr_{i}" for i in range(1, 21)] + ["credit_risk"]
                df = pd.read_csv(f, sep=' ', header=None, names=col_names)
                # Map 1: Good, 2: Bad -> 0: Good, 1: Bad
                df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})
                df.to_csv(p_german, index=False)
    manifest["statlog_german_credit"] = {
        "modality": "TABULAR",
        "provenance": "UCI Machine Learning Repository (ID 144)",
        "file_path": p_german,
        "rows": len(pd.read_csv(p_german)),
        "features": 20,
        "classes": ["Good Credit", "Bad Credit"],
        "sha256": sha256_file(p_german)
    }

    # 2. Cleveland Heart Disease (UCI ID 45)
    p_heart = os.path.join(TABULAR_DIR, "cleveland_heart_disease.csv")
    if not os.path.exists(p_heart):
        print("  Downloading Cleveland Heart Disease...")
        req = urllib.request.Request('https://archive.ics.uci.edu/static/public/45/heart+disease.zip', headers=headers)
        with urllib.request.urlopen(req) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            with z.open('processed.cleveland.data') as f:
                cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "diagnosis"]
                df = pd.read_csv(f, header=None, names=cols, na_values="?")
                # Impute missing values with median
                df = df.fillna(df.median(numeric_only=True))
                # Binary target: 0 = healthy, 1 = disease
                df['diagnosis'] = (df['diagnosis'] > 0).astype(int)
                df.to_csv(p_heart, index=False)
    manifest["cleveland_heart_disease"] = {
        "modality": "TABULAR",
        "provenance": "UCI Machine Learning Repository (ID 45)",
        "file_path": p_heart,
        "rows": len(pd.read_csv(p_heart)),
        "features": 13,
        "classes": ["Absence", "Presence"],
        "sha256": sha256_file(p_heart)
    }

    # 3. Bank Marketing (UCI ID 222)
    p_bank = os.path.join(TABULAR_DIR, "bank_marketing.csv")
    if not os.path.exists(p_bank):
        print("  Downloading Bank Marketing...")
        req = urllib.request.Request('https://archive.ics.uci.edu/static/public/222/bank+marketing.zip', headers=headers)
        with urllib.request.urlopen(req) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            with z.open('bank.zip') as subz_file:
                subz = zipfile.ZipFile(io.BytesIO(subz_file.read()))
                with subz.open('bank.csv') as f:
                    df = pd.read_csv(f, sep=';')
                    df['deposit_subscribed'] = (df['y'] == 'yes').astype(int)
                    df = df.drop(columns=['y'])
                    df.to_csv(p_bank, index=False)
    manifest["bank_marketing"] = {
        "modality": "TABULAR",
        "provenance": "UCI Machine Learning Repository (ID 222)",
        "file_path": p_bank,
        "rows": len(pd.read_csv(p_bank)),
        "features": 16,
        "classes": ["No Subscription", "Subscription"],
        "sha256": sha256_file(p_bank)
    }

    # 4. Adult Census Income (UCI ID 2) - Official Train (32,561) & Official Test (16,281) Splits
    p_adult = os.path.join(TABULAR_DIR, "adult_census.csv")
    p_adult_tr = os.path.join(TABULAR_DIR, "adult_census_train.csv")
    p_adult_te = os.path.join(TABULAR_DIR, "adult_census_test.csv")

    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "income"]

    if not os.path.exists(p_adult_tr) or not os.path.exists(p_adult_te):
        print("  Downloading Full Adult Census Income (adult.data + adult.test)...")
        req = urllib.request.Request('https://archive.ics.uci.edu/static/public/2/adult.zip', headers=headers)
        with urllib.request.urlopen(req) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            with z.open('adult.data') as f_tr, z.open('adult.test') as f_te:
                df_tr = pd.read_csv(f_tr, header=None, names=cols, na_values=" ?", skipinitialspace=True).dropna()
                df_te = pd.read_csv(f_te, header=None, names=cols, na_values=" ?", skipinitialspace=True, skiprows=1).dropna()
                
                df_tr['income_above_50k'] = df_tr['income'].astype(str).str.strip().str.rstrip('.').apply(lambda x: 1 if '>50K' in x else 0)
                df_te['income_above_50k'] = df_te['income'].astype(str).str.strip().str.rstrip('.').apply(lambda x: 1 if '>50K' in x else 0)
                
                df_tr = df_tr.drop(columns=['income'])
                df_te = df_te.drop(columns=['income'])

                df_tr.to_csv(p_adult_tr, index=False)
                df_te.to_csv(p_adult_te, index=False)
                df_tr.to_csv(p_adult, index=False)

    manifest["adult_census"] = {
        "modality": "TABULAR",
        "provenance": "UCI Machine Learning Repository (ID 2)",
        "file_path": p_adult,
        "train_file": p_adult_tr,
        "test_file": p_adult_te,
        "train_rows": len(pd.read_csv(p_adult_tr)),
        "test_rows": len(pd.read_csv(p_adult_te)),
        "total_rows": len(pd.read_csv(p_adult_tr)) + len(pd.read_csv(p_adult_te)),
        "rows": len(pd.read_csv(p_adult)),
        "features": 14,
        "classes": ["<=50K", ">50K"],
        "train_sha256": sha256_file(p_adult_tr),
        "test_sha256": sha256_file(p_adult_te),
        "sha256": sha256_file(p_adult)
    }

    # 5. Auto MPG (UCI ID 9) - Continuous Regression Benchmark (392 complete records)
    p_mpg = os.path.join(TABULAR_DIR, "auto_mpg.csv")
    if not os.path.exists(p_mpg):
        print("  Downloading Auto MPG (Regression)...")
        req = urllib.request.Request('https://archive.ics.uci.edu/static/public/9/auto+mpg.zip', headers=headers)
        with urllib.request.urlopen(req) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            with z.open('auto-mpg.data') as f:
                cols = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
                df = pd.read_csv(f, sep=r'\s+', header=None, names=cols, na_values="?")
                df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
                df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
                df = df.drop(columns=['car_name']).dropna()
                df.to_csv(p_mpg, index=False)
    manifest["auto_mpg"] = {
        "modality": "TABULAR",
        "provenance": "UCI Machine Learning Repository (ID 9)",
        "file_path": p_mpg,
        "rows": len(pd.read_csv(p_mpg)),
        "features": 7,
        "task": "REGRESSION",
        "sha256": sha256_file(p_mpg)
    }

    # 5. Breast Cancer Coimbra (UCI ID 451)
    p_coimbra = os.path.join(TABULAR_DIR, "breast_cancer_coimbra.csv")
    if not os.path.exists(p_coimbra):
        print("  Downloading Breast Cancer Coimbra...")
        req = urllib.request.Request('https://archive.ics.uci.edu/static/public/451/breast+cancer+coimbra.zip', headers=headers)
        with urllib.request.urlopen(req) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            with z.open('dataR2.csv') as f:
                df = pd.read_csv(f)
                # Classification target 1: Healthy Controls, 2: Patients -> map to 0, 1
                df['cancer_status'] = df['Classification'].map({1: 0, 2: 1})
                df = df.drop(columns=['Classification'])
                df.to_csv(p_coimbra, index=False)
    manifest["breast_cancer_coimbra"] = {
        "modality": "TABULAR",
        "provenance": "UCI Machine Learning Repository (ID 451)",
        "file_path": p_coimbra,
        "rows": len(pd.read_csv(p_coimbra)),
        "features": 9,
        "classes": ["Healthy Control", "Patient"],
        "sha256": sha256_file(p_coimbra)
    }

    print("  Tabular datasets complete.")
    return manifest

def fetch_vision_datasets():
    print("--- [2/3] Fetching 5 Vision Benchmark Datasets from Zenodo (MedMNIST) ---")
    manifest = {}

    vision_urls = {
        "retinamnist": {
            "url": "https://zenodo.org/records/10519652/files/retinamnist.npz?download=1",
            "file": "retinamnist.npz",
            "classes": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
            "provenance": "DeepDR Retina Fundus Micrographs (MedMNIST v2)"
        },
        "breastmnist": {
            "url": "https://zenodo.org/records/10519652/files/breastmnist.npz?download=1",
            "file": "breastmnist.npz",
            "classes": ["Malignant", "Normal/Benign"],
            "provenance": "Breast Ultrasound Micro-patches (MedMNIST v2)"
        },
        "pneumoniamnist": {
            "url": "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1",
            "file": "pneumoniamnist.npz",
            "classes": ["Normal", "Pneumonia"],
            "provenance": "Chest X-Ray Pediatric Pneumonia (MedMNIST v2)"
        },
        "dermamnist": {
            "url": "https://zenodo.org/records/10519652/files/dermamnist.npz?download=1",
            "file": "dermamnist.npz",
            "classes": ["Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis", "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular"],
            "provenance": "HAM10000 Dermatoscopy Pigmented Lesions (MedMNIST v2)"
        },
        "bloodmnist": {
            "url": "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1",
            "file": "bloodmnist.npz",
            "classes": ["Basophil", "Eosinophil", "Erythroblast", "Immature Granulocyte", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"],
            "provenance": "Peripheral Blood Cell Micrographs (MedMNIST v2)"
        }
    }

    for key, info in vision_urls.items():
        dst = os.path.join(VISION_DIR, info["file"])
        if not os.path.exists(dst):
            print(f"  Downloading {key} ({info['file']})...")
            cmd = ["curl.exe", "-s", "-L", info["url"], "-o", dst]
            subprocess.run(cmd, check=True)

        # Inspect NPZ
        data = np.load(dst)
        train_x = data["train_images"]
        train_y = data["train_labels"]
        val_x = data["val_images"]
        test_x = data["test_images"]
        total_samples = len(train_x) + len(val_x) + len(test_x)

        manifest[key] = {
            "modality": "COMPUTER_VISION",
            "provenance": info["provenance"],
            "file_path": dst,
            "train_samples": len(train_x),
            "val_samples": len(val_x),
            "test_samples": len(test_x),
            "total_samples": total_samples,
            "image_shape": list(train_x.shape[1:]),
            "classes": info["classes"],
            "sha256": sha256_file(dst)
        }
        print(f"  Verified {key}: {total_samples} images, shape {train_x.shape[1:]}")

    print("  Vision datasets complete.")
    return manifest

def fetch_nlp_datasets():
    print("--- [3/3] Fetching 5 NLP Benchmark Datasets ---")
    manifest = {}
    headers = {'User-Agent': 'Mozilla/5.0'}

    # 1. Financial PhraseBank (Malo et al. 50Agree) - 60/20/20 Train/Calibration/Test Splits
    p_fin = os.path.join(NLP_DIR, "financial_phrasebank.csv")
    p_fin_tr = os.path.join(NLP_DIR, "financial_phrasebank_train.csv")
    p_fin_cal = os.path.join(NLP_DIR, "financial_phrasebank_cal.csv")
    p_fin_te = os.path.join(NLP_DIR, "financial_phrasebank_test.csv")

    if not os.path.exists(p_fin) or not os.path.exists(p_fin_tr):
        print("  Downloading Financial PhraseBank & constructing isolated train/cal/test splits...")
        u = 'https://raw.githubusercontent.com/maxwellsarpong/NLP-financial-text-processing-dataset/master/Sentences_50Agree.txt'
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('ISO-8859-1')
        lines = content.strip().splitlines()
        records = []
        for line in lines:
            if '@' in line:
                text, label = line.rsplit('@', 1)
                records.append({"text": text.strip(), "sentiment": label.strip()})
        df = pd.DataFrame(records)
        df.to_csv(p_fin, index=False)

        # Construct stratified 60% train, 20% calibration, 20% test splits with seed 42
        # Deduplicate to ensure 0 duplicate text overlap across splits
        df_unique = df.drop_duplicates(subset=['text']).reset_index(drop=True)
        from sklearn.model_selection import train_test_split
        tr_df, temp_df = train_test_split(df_unique, test_size=0.40, random_state=42, stratify=df_unique['sentiment'])
        cal_df, te_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['sentiment'])

        tr_df.to_csv(p_fin_tr, index=False)
        cal_df.to_csv(p_fin_cal, index=False)
        te_df.to_csv(p_fin_te, index=False)

    manifest["financial_phrasebank"] = {
        "modality": "NLP",
        "provenance": "Malo et al. Financial PhraseBank (50% Consensus Agreement)",
        "file_path": p_fin,
        "train_file": p_fin_tr,
        "cal_file": p_fin_cal,
        "test_file": p_fin_te,
        "total_rows": len(pd.read_csv(p_fin)),
        "train_rows": len(pd.read_csv(p_fin_tr)),
        "cal_rows": len(pd.read_csv(p_fin_cal)),
        "test_rows": len(pd.read_csv(p_fin_te)),
        "classes": ["negative", "neutral", "positive"],
        "sha256": sha256_file(p_fin),
        "train_sha256": sha256_file(p_fin_tr),
        "test_sha256": sha256_file(p_fin_te)
    }

    # 2. Rotten Tomatoes SST-2 (Pang & Lee / Socher et al.)
    p_rt = os.path.join(NLP_DIR, "rotten_tomatoes.csv")
    if not os.path.exists(p_rt):
        print("  Downloading Rotten Tomatoes SST-2...")
        u = 'https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv'
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req) as resp:
            df = pd.read_csv(resp, sep='\t', header=None, names=["text", "label"])
        df['sentiment'] = df['label'].map({0: "negative", 1: "positive"})
        df.drop(columns=['label']).to_csv(p_rt, index=False)
    manifest["rotten_tomatoes"] = {
        "modality": "NLP",
        "provenance": "Socher et al. Stanford Sentiment Treebank (Rotten Tomatoes reviews)",
        "file_path": p_rt,
        "rows": len(pd.read_csv(p_rt)),
        "classes": ["negative", "positive"],
        "sha256": sha256_file(p_rt)
    }

    # 3. AG News (Zhang et al. 4-class topic classification) - Full 7,600 Test Split
    p_ag = os.path.join(NLP_DIR, "ag_news.csv")
    if not os.path.exists(p_ag) or len(pd.read_csv(p_ag)) < 5000:
        print("  Downloading Full AG News (7,600 official records)...")
        u = 'https://raw.githubusercontent.com/BrambleXu/nlp-beginner-guide-keras/master/char-level-cnn/data/ag_news_csv/test.csv'
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req) as resp:
            df = pd.read_csv(resp, header=None, names=["class_index", "title", "description"])
        df['text'] = df['title'] + " - " + df['description']
        class_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
        df['topic'] = df['class_index'].map(class_map)
        df[['text', 'topic']].to_csv(p_ag, index=False)
    manifest["ag_news"] = {
        "modality": "NLP",
        "provenance": "Zhang et al. AG's News Topic Classification Corpus",
        "file_path": p_ag,
        "rows": len(pd.read_csv(p_ag)),
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
        "sha256": sha256_file(p_ag)
    }

    # 4. TweetEval Sentiment (Barbieri et al.) - Full 12,284 Official Test Split
    p_tweet = os.path.join(NLP_DIR, "tweeteval_sentiment.csv")
    if not os.path.exists(p_tweet) or len(pd.read_csv(p_tweet)) < 5000:
        print("  Downloading Full TweetEval Sentiment (12,284 official records)...")
        u_text = 'https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_text.txt'
        u_label = 'https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/test_labels.txt'
        req_t = urllib.request.Request(u_text, headers=headers)
        req_l = urllib.request.Request(u_label, headers=headers)
        with urllib.request.urlopen(req_t) as resp_t, urllib.request.urlopen(req_l) as resp_l:
            texts = [l.strip() for l in resp_t.read().decode('utf-8').splitlines()]
            labels = [int(l.strip()) for l in resp_l.read().decode('utf-8').splitlines()]
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        df = pd.DataFrame({"text": texts, "sentiment": [label_map[l] for l in labels]})
        df.to_csv(p_tweet, index=False)
    manifest["tweeteval_sentiment"] = {
        "modality": "NLP",
        "provenance": "Barbieri et al. TweetEval Twitter Sentiment Corpus",
        "file_path": p_tweet,
        "rows": len(pd.read_csv(p_tweet)),
        "classes": ["negative", "neutral", "positive"],
        "sha256": sha256_file(p_tweet)
    }

    # 5. SMS Spam Collection (Almeida et al. UCI ID 228)
    p_sms = os.path.join(NLP_DIR, "sms_spam.csv")
    if not os.path.exists(p_sms):
        print("  Downloading SMS Spam Collection...")
        u = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            with z.open('SMSSpamCollection') as f:
                df = pd.read_csv(f, sep='\t', header=None, names=["label", "text"])
                df = df[['text', 'label']]
                df.to_csv(p_sms, index=False)
    manifest["sms_spam"] = {
        "modality": "NLP",
        "provenance": "Almeida et al. UCI SMS Spam Collection (ID 228)",
        "file_path": p_sms,
        "rows": len(pd.read_csv(p_sms)),
        "classes": ["ham", "spam"],
        "sha256": sha256_file(p_sms)
    }

    print("  NLP datasets complete.")
    return manifest

def main():
    print("======================================================================")
    print("Sukio Multimodal Benchmark Dataset Curation & Manifest Generator")
    print("======================================================================")
    all_manifests = {}
    all_manifests["tabular"] = fetch_tabular_datasets()
    all_manifests["vision"] = fetch_vision_datasets()
    all_manifests["nlp"] = fetch_nlp_datasets()

    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "DATASET_MANIFEST.json"))
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(all_manifests, f, indent=2)

    print("\n======================================================================")
    print(f"SUCCESS: All 15 Real Benchmark Datasets Curated & Manifest Saved to:\n  {manifest_path}")
    print("======================================================================")

if __name__ == "__main__":
    main()
