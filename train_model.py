import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from model_pipeline import UnifiedPhishingPipeline

def train_and_save_unified_model():
    """Train and save complete pipeline as single .pkl"""
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('data/url_dataset.csv')
    df['url'] = df['url'].astype(str).str.strip()
    df['type'] = df['type'].astype(str).str.strip().str.lower()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['type'].value_counts()}")
    
    # Split data
    X = df['url'].values
    y = df['type'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\nTraining size: {X_train.shape[0]}")
    print(f"Testing size: {X_test.shape[0]}")
    
    # Train unified pipeline
    print("\nTraining unified pipeline...")
    pipeline = UnifiedPhishingPipeline(model_type='ensemble')
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    y_pred = pipeline.predict(X_test)
    y_test_binary = (y_test == 'phishing').astype(int)
    y_pred_binary = (np.array(y_pred) == 'phishing').astype(int)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test_binary, y_pred_binary):.4f}")
    print(f"Precision: {precision_score(y_test_binary, y_pred_binary):.4f}")
    print(f"Recall: {recall_score(y_test_binary, y_pred_binary):.4f}")
    print(f"F1-Score: {f1_score(y_test_binary, y_pred_binary):.4f}")
    
    # Test single prediction
    print("\nTesting single URL prediction...")
    test_urls = [
        "https://www.wikipedia.org",
        "http://verify-paypal-account-secure-login.com",
        "http://bit.ly/secure-banking-update"
    ]
    
    for url in test_urls:
        proba = pipeline.predict_proba([url])[0]
        pred = pipeline.predict([url])[0]
        print(f"\nURL: {url}")
        print(f"Prediction: {pred}")
        print(f"Legitimate prob: {proba[0]:.3f}, Phishing prob: {proba[1]:.3f}")
    
    # Save complete pipeline as single .pkl
    pipeline.save('models/phishing_detector.pkl')
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("Complete pipeline saved as: models/phishing_detector.pkl")
    print("This single file contains both feature extraction and model.")
    print("="*50)
    
    # Also save a smaller version for demo
    sample_urls = [
        "https://www.google.com",
        "http://secure-login-verify-account.com",
        "https://www.paypal.com"
    ]
    sample_df = pd.DataFrame({'url': sample_urls})
    sample_df.to_csv('data/sample_urls.csv', index=False)

if __name__ == "__main__":
    train_and_save_unified_model()