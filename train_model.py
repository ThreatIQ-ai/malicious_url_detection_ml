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
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    
    y_pred = pipeline.predict(X_test)
    y_test_binary = (y_test == 'phishing').astype(int)
    y_pred_binary = (np.array(y_pred) == 'phishing').astype(int)
    
    print("\n" + "="*50)
    print("Model Performance on Test Set:")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test_binary, y_pred_binary):.4f}")
    print(f"Precision: {precision_score(y_test_binary, y_pred_binary):.4f}")
    print(f"Recall: {recall_score(y_test_binary, y_pred_binary):.4f}")
    print(f"F1-Score: {f1_score(y_test_binary, y_pred_binary):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_binary, y_pred_binary, 
                               target_names=['legitimate', 'phishing']))
    
    # Test single prediction with error handling
    print("\n" + "="*50)
    print("Testing single URL prediction...")
    print("="*50)
    
    test_urls = [
        "https://www.wikipedia.org",
        "http://verify-paypal-account-secure-login.com",
        "http://bit.ly/secure-banking-update",
        "https://example.com/test.php",
        "https://example.com/index.html",
        "http://192.168.1.1/login.php"
    ]
    
    for url in test_urls:
        try:
            proba = pipeline.predict_proba([url])[0]
            pred = pipeline.predict([url])[0]
            print(f"\nURL: {url}")
            print(f"Prediction: {pred}")
            print(f"Legitimate prob: {proba[0]:.3f}, Phishing prob: {proba[1]:.3f}")
            
            # Show some extracted features
            feature_extractor = pipeline.pipeline.named_steps['feature_extractor']
            features = feature_extractor.transform([url])
            print(f"Has file extension: {features['has_file_extension'].iloc[0]}")
            print(f"Is PHP: {features['is_php'].iloc[0]}, Is HTML: {features['is_html'].iloc[0]}")
            
        except Exception as e:
            print(f"\nError processing URL '{url}': {str(e)}")
    
    # Save complete pipeline as single .pkl
    pipeline.save('models/phishing_detector.pkl')
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("Complete pipeline saved as: models/phishing_detector.pkl")
    print("This single file contains both feature extraction and model.")
    print("="*50)
    
    # Test loading and using the saved model
    print("\nTesting saved model load...")
    try:
        loaded_pipeline = joblib.load('models/phishing_detector.pkl')
        test_result = loaded_pipeline.predict(["https://www.google.com"])
        print(f"✓ Model loaded successfully. Test prediction: {test_result[0]}")
    except Exception as e:
        print(f"✗ Error loading saved model: {e}")

if __name__ == "__main__":
    train_and_save_unified_model()