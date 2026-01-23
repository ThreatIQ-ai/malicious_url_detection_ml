# Real-Time URL Detection API  
### Supervised Multi-Class Machine Learning System

---
## Dataset Description

The dataset contains a total of **651,191 URLs**, labeled for supervised multi-class classification.  
Each URL is categorized based on its level of maliciousness.

### Class Distribution
- **Benign (Safe):** 428,103 URLs  
- **Defacement:** 96,457 URLs  
- **Phishing:** 94,111 URLs  
- **Malware:** 32,520 URLs  

### Dataset Structure
The dataset consists of **two columns**:

| Column Name | Description |
|------------|------------|
| `url` | Raw URL string |
| `type` | Class label indicating maliciousness (`benign`, `defacement`, `phishing`, `malware`) |

### Key Observations
- The dataset is **imbalanced**, with benign URLs forming the majority class.
- Malicious classes exhibit distinct but overlapping patterns, requiring robust feature engineering.
- The dataset size is sufficient for training high-capacity supervised machine learning models.

This dataset serves as the foundation for building and evaluating the real-time URL detection system.


---

## 1. Problem Definition & Requirements

### 1.1 Goal
Design and implement a **real-time URL detection API** capable of classifying URLs into the following categories:

- **Benign**
- **Phishing**
- **Malware**
- **Defacement**

The system must provide fast, accurate predictions suitable for real-time security applications such as browsers, firewalls, and SOC pipelines.

---

## 2. Data Ingestion & Understanding

### 2.1 Dataset Overview
- **Input Features:** Raw URL string  
- **Target Variable:** URL type (benign, phishing, malware, defacement)
- **Dataset Size:** ~650,000 URLs
- **Learning Type:** Supervised Learning
- **Task Type:** Multi-Class Classification

Raw data consists of only two columns (`url`, `type`), requiring extensive preprocessing and feature engineering.

---

## 3. Data Preprocessing Pipeline

### A. Data Cleaning & URL Normalization

#### 3.1 URL Canonicalization
To ensure consistency and reduce noise, the following normalization steps are applied:

- Convert URLs to lowercase
- Strip leading and trailing whitespaces
- Remove URL fragments (e.g., `#section`)
- Decode percent-encoded characters (e.g., `%20`)
- Normalize redundant slashes (`//`)

#### 3.2 Invalid URL Handling
- Remove malformed URLs
- Filter URLs missing scheme or domain
- Deduplicate identical URLs

**Purpose:**  
Improve feature reliability and prevent downstream extraction errors.

---

### B. URL Parsing & Component Extraction

URLs are parsed into structured components using a URL parser.

| Component  | Example |
|-----------|--------|
| Scheme    | `http`, `https` |
| Domain    | `example.com` |
| Subdomain | `login.example.com` |
| Path      | `/verify/account` |
| Query     | `?id=123` |
| Fragment  | `#section` |

These components form the foundation for feature engineering.

---

## 4. Feature Engineering Strategy (≈ 50 Features)

The objective is to transform raw URLs into a **compact, informative feature set** optimized for real-time inference.

---

### 4.1 Lexical (Character-Based) Features

Capture how the URL *looks* at a character level.

| Feature | Rationale |
|------|----------|
| URL length | Malicious URLs tend to be longer |
| Domain length | Abnormally long domains indicate risk |
| Path length | Phishing URLs often embed long paths |
| Digit count | Numeric obfuscation is common |
| Special character count (`-`, `_`, `@`, `?`, `=`) | Obfuscation indicator |
| Digit-to-character ratio | Randomness detection |
| Dot (`.`) count | Excessive subdomains |

---

### 4.2 Structural Features

Describe URL composition and hierarchy.

| Feature | Description |
|------|------------|
| Number of subdomains | Multiple subdomains increase risk |
| IP address as domain | Common malware behavior |
| URL depth (`/` count) | Deeper paths are suspicious |
| Query parameter count | Phishing often uses parameters |
| URL shortening service | Used to hide malicious intent |

---

### 4.3 Security & Protocol Features (Binary)

Simple but highly predictive indicators.

| Feature | Type |
|------|------|
| HTTPS present | Binary |
| HTTP used | Binary |
| Non-standard port | Binary |
| URL redirection (`//`) | Binary |
| Encoded characters present | Binary |

---

### 4.4 Suspicious Keyword Features

Detect known social engineering and malware patterns.

**Keyword Categories:**
- **Authentication:** `login`, `signin`, `verify`
- **Financial:** `bank`, `payment`, `invoice`
- **Urgency:** `secure`, `update`, `confirm`
- **Malware Signals:** `exe`, `download`, `apk`

**Generated Features:**
- Presence of any suspicious keyword (binary)
- Count of suspicious keywords
- Category-wise keyword flags

---

### 4.5 Domain-Based Features (Offline / Cached)

Computed offline or cached to avoid real-time latency.

| Feature | Description |
|------|------------|
| TLD risk level (`.tk`, `.ru`, `.xyz`) | Known abuse zones |
| Domain length anomaly | Statistical deviation |
| Domain entropy | Random-looking domains |
| Brand impersonation detection | `paypa1`, `goog1e` |

---

## 5. Model Selection Strategy

### 5.1 Problem Restatement
- **Learning Type:** Supervised
- **Task:** Multi-Class Classification (4 classes)
- **Input:** ~50 engineered numerical & binary features
- **Constraints:**
  - Low latency inference
  - High precision for malicious classes
  - Scalability to large datasets
  - Robustness to noisy and imbalanced data

---

## 6. Recommended Models

### 6.1 Gradient Boosted Decision Trees (⭐ Best Overall)
**Examples:** XGBoost, LightGBM, CatBoost

#### Why They Are Ideal

| Reason | Explanation |
|----|----|
| Non-linear learning | URL attacks are irregular |
| Tabular data performance | Designed for structured features |
| Minimal scaling required | Tree splits are scale-invariant |
| Native multiclass support | Softmax objective |
| Outlier robustness | Common in URL statistics |
| Feature importance | Security explainability |
| High accuracy | Industry-proven |

**Model Selection Guidance:**
- **LightGBM:** Best speed & scalability
- **XGBoost:** Strong stability & tuning
- **CatBoost:** Best with categorical features

📌 **Verdict:**  
**Primary production model**

---

### 6.2 Random Forest (Strong Baseline)

**Advantages:**
- Ensemble generalization
- Natural multiclass handling
- Overfitting resistance
- No feature scaling required

**Limitations:**
- Slower inference
- Larger memory footprint
- Lower accuracy than boosting

📌 **Verdict:**  
**Baseline and fallback model**

---

### 6.3 Logistic Regression (One-vs-Rest)

**Advantages:**
- Fast training & inference
- Probabilistic outputs
- Good benchmarking model

**Limitations:**
- Linear decision boundaries
- Poor complex pattern learning
- Feature-quality sensitive

📌 **Verdict:**  
**Benchmark only**

---

## 7. Models Not Recommended

---

### 7.1 Support Vector Machines (SVM)

Support Vector Machines are **not recommended** for this problem due to scalability and performance limitations.

**Why SVM is not suitable:**
- **Poor scalability:** Training SVMs on large datasets (650k+ URLs) is very slow.
- **High memory usage:** Kernel-based SVMs require storing large matrices in memory.
- **Multi-class complexity:** Requires One-vs-One (OVO) or One-vs-Rest (OVR) strategies, increasing computation.
- **Slow inference:** Not ideal for real-time API responses.

**Conclusion:**  
SVMs are impractical for large-scale, real-time URL classification systems.

---

### 7.2 k-Nearest Neighbors (kNN)

kNN is a **distance-based algorithm** that performs poorly in real-time environments.

**Why kNN is not suitable:**
- **Extremely slow predictions:** Every new URL must be compared with the entire training dataset.
- **High memory requirement:** The full dataset must be stored in memory.
- **Sensitive to noise:** URL features are often noisy and high-dimensional.
- **No learning phase:** Does not generalize patterns, only memorizes data.

**Conclusion:**  
kNN cannot meet real-time performance or scalability requirements.

---

### 7.3 Naive Bayes

Naive Bayes is a **simple probabilistic classifier**, but it is too weak for this task.

**Why Naive Bayes is not suitable:**
- **Unrealistic assumptions:** Assumes all features are independent, which is not true for URL data.
- **Low expressive power:** Cannot capture complex attack patterns.
- **High false positives:** Especially poor at separating phishing and malware URLs.
- **Limited adaptability:** Performs poorly on evolving threats.

**Conclusion:**  
Naive Bayes is too simplistic for accurate URL threat detection.

---

### 7.4 Rule-Based or Regex-Based Systems

Rule-based approaches rely on manually written rules or regular expressions instead of learning patterns.

**Why rule-based systems are not suitable:**
- **Easy to bypass:** Attackers can slightly modify URLs to evade rules.
- **Hard to maintain:** Rules grow rapidly and become unmanageable.
- **Poor generalization:** Cannot detect previously unseen attacks.
- **Not machine learning:** No learning or adaptation capability.

**Conclusion:**  
Rule-based systems may assist as a secondary filter but **cannot replace ML-based detection**.

---


## 8. Final Recommendation

> **Gradient Boosted Decision Trees (LightGBM/XGBoost)** provide the best balance of accuracy, speed, scalability, and interpretability for real-time multi-class URL threat detection.

---

