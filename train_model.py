import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from model_pipeline import URLFeatureExtractor, ModelSelector, UnifiedPhishingPipeline

def prepare_sample_data(sample_size=50000):
    """Load and prepare sample data for model selection"""
    print("📊 Loading dataset for model selection...")
    df = pd.read_csv('URLShield/data/url_dataset.csv')
    df['url'] = df['url'].astype(str).str.strip()
    df['type'] = df['type'].astype(str).str.strip().str.lower()
    
    print(f"📈 Full dataset shape: {df.shape}")
    print(f"📊 Class distribution in full dataset:")
    print(f"   Legitimate: {(df['type'] == 'legitimate').sum()} ({(df['type'] == 'legitimate').mean()*100:.2f}%)")
    print(f"   Phishing: {(df['type'] == 'phishing').sum()} ({(df['type'] == 'phishing').mean()*100:.2f}%)")
    
    # Take stratified sample for model selection
    print(f"\n🔍 Taking stratified sample of {sample_size} for model selection...")
    df_sample = df.groupby('type', group_keys=False).apply(
        lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))), random_state=42)
    ).reset_index(drop=True)
    
    print(f"📦 Sample dataset shape: {df_sample.shape}")
    print(f"📊 Class distribution in sample:")
    print(f"   Legitimate: {(df_sample['type'] == 'legitimate').sum()} ({(df_sample['type'] == 'legitimate').mean()*100:.2f}%)")
    print(f"   Phishing: {(df_sample['type'] == 'phishing').sum()} ({(df_sample['type'] == 'phishing').mean()*100:.2f}%)")
    
    # Split sample data
    X_sample = df_sample['url'].values
    y_sample = df_sample['type'].values
    
    # Split into train and validation sets (for model selection)
    X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
        X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
    )
    
    print(f"\n📋 Sample Data Split:")
    print(f"   Training samples: {X_train_sample.shape[0]}")
    print(f"   Validation samples: {X_val_sample.shape[0]}")
    
    return X_train_sample, X_val_sample, y_train_sample, y_val_sample, df

def prepare_full_data(df):
    """Prepare full dataset for final training"""
    print("\n📊 Preparing full dataset for final training...")
    
    X_full = df['url'].values
    y_full = df['type'].values
    
    # Split full data into train and test sets
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=0.15, stratify=y_full, random_state=42
    )
    
    print(f"📋 Full Data Split:")
    print(f"   Training samples: {X_train_full.shape[0]}")
    print(f"   Testing samples: {X_test_full.shape[0]}")
    
    return X_train_full, X_test_full, y_train_full, y_test_full

def extract_features(X_train, X_val, X_test=None):
    """Extract features from URLs"""
    print("\n🔧 Extracting features...")
    start_time = time.time()
    
    feature_extractor = URLFeatureExtractor()
    
    # Fit on training data
    feature_extractor.fit(X_train)
    
    # Transform all datasets
    X_train_features = feature_extractor.transform(X_train)
    X_val_features = feature_extractor.transform(X_val)
    
    if X_test is not None:
        X_test_features = feature_extractor.transform(X_test)
    else:
        X_test_features = None
    
    elapsed_time = time.time() - start_time
    print(f"✅ Feature extraction completed in {elapsed_time:.2f} seconds")
    print(f"📊 Number of features: {X_train_features.shape[1]}")
    
    return X_train_features, X_val_features, X_test_features, feature_extractor

def select_best_model_on_sample(X_train_sample_features, y_train_sample_binary, 
                                X_val_sample_features, y_val_sample_binary):
    """Select best model using sample data"""
    print("\n" + "="*70)
    print("🔬 MODEL SELECTION PHASE (Using Sample Data)")
    print("="*70)
    
    selector = ModelSelector()
    
    # Train and evaluate models on sample data
    start_time = time.time()
    selector.train_and_evaluate(
        X_train_sample_features, y_train_sample_binary,
        X_val_sample_features, y_val_sample_binary
    )
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Model selection completed in {elapsed_time:.2f} seconds")
    
    # Print comparison
    selector.print_model_comparison()
    
    # Get best model info
    best_model, best_model_name = selector.get_best_model_info()
    
    if best_model is None:
        raise ValueError("No valid model found during selection!")
    
    return best_model, best_model_name, selector.get_all_results()

def train_final_model_on_full_data(best_model, best_model_name, feature_extractor, 
                                  X_train_full, y_train_full_binary,
                                  X_test_full, y_test_full_binary):
    """Train the selected best model on full dataset"""
    print("\n" + "="*70)
    print(f"🚀 FINAL TRAINING PHASE (Training {best_model_name} on Full Data)")
    print("="*70)
    
    print(f"📊 Data shapes:")
    print(f"   X_train_full: {X_train_full.shape[0]} URLs")
    print(f"   y_train_full_binary: {y_train_full_binary.shape}")
    print(f"   X_test_full: {X_test_full.shape[0]} URLs")
    print(f"   y_test_full_binary: {y_test_full_binary.shape}")
    
    # Create complete pipeline with the best model's classifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    # Extract the classifier from the best model pipeline
    if hasattr(best_model, 'named_steps'):
        classifier = best_model.named_steps['classifier']
    else:
        classifier = best_model
    
    print(f"\n🔧 Creating final pipeline with {best_model_name} classifier...")
    
    # Create the complete pipeline for deployment
    final_pipeline = Pipeline([
        ('feature_extractor', feature_extractor),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    # Train on full data
    print(f"📈 Training on {len(X_train_full):,} samples...")
    start_time = time.time()
    
    final_pipeline.fit(X_train_full, y_train_full_binary)
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    y_pred = final_pipeline.predict(X_test_full)
    y_pred_proba = final_pipeline.predict_proba(X_test_full)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_full_binary, y_pred)
    precision = precision_score(y_test_full_binary, y_pred, zero_division=0)
    recall = recall_score(y_test_full_binary, y_pred, zero_division=0)
    f1 = f1_score(y_test_full_binary, y_pred, zero_division=0)
    auc = roc_auc_score(y_test_full_binary, y_pred_proba)
    
    print(f"\n🎯 Final Model Performance on Test Set:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC:       {auc:.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test_full_binary, y_pred, 
                               target_names=['legitimate', 'phishing']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_full_binary, y_pred)
    print("\n📊 Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Legitimate  Phishing")
    print(f"Actual Legitimate    {cm[0,0]:>7}     {cm[0,1]:>7}")
    print(f"Actual Phishing      {cm[1,0]:>7}     {cm[1,1]:>7}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📈 Additional Metrics:")
    print(f"   True Negatives (TN):  {tn}")
    print(f"   False Positives (FP): {fp}")
    print(f"   False Negatives (FN): {fn}")
    print(f"   True Positives (TP):  {tp}")
    print(f"   False Positive Rate:  {fp/(fp+tn):.4f}")
    print(f"   False Negative Rate:  {fn/(fn+tp):.4f}")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Phishing'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.tight_layout()
    plt.savefig('URLShield/models/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Confusion matrix saved: URLShield/models/confusion_matrix.png")
    
    return UnifiedPhishingPipeline.from_pipeline(final_pipeline), {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }

def test_single_predictions(pipeline):
    """Test the pipeline with single URLs"""
    print("\n" + "="*70)
    print("🧪 TESTING SINGLE URL PREDICTIONS")
    print("="*70)
    
    test_urls = [
        # Legitimate URLs
        ("https://www.wikipedia.org", "Legitimate"),
        ("https://www.google.com", "Legitimate"),
        ("https://github.com", "Legitimate"),
        ("https://stackoverflow.com", "Legitimate"),
        ("https://www.linkedin.com", "Legitimate"),
        
        # Phishing URLs
        ("http://verify-paypal-account-secure-login.com", "Phishing"),
        ("http://bit.ly/secure-banking-update", "Phishing"),
        ("https://example.com/test.php?user=admin&password=123", "Phishing"),
        ("http://192.168.1.1/login.php", "Phishing"),
        ("https://secure-login-bank-account-verification-update-now.com", "Phishing"),
        
        # Edge cases
        ("https://example.com/index.html", "Legitimate"),
        ("http://paypal-verify-account-limited-urgent-secure-alert.com/login", "Phishing"),
    ]
    
    print("\n" + "-"*90)
    print(f"{'URL':<50} {'Expected':<12} {'Predicted':<12} {'Phishing Prob':<12} {'Risk':<10}")
    print("-"*90)
    
    for url, expected in test_urls:
        try:
            proba = pipeline.predict_proba([url])[0]
            pred = pipeline.predict([url])[0]
            
            phishing_prob = proba[1]
            if phishing_prob < 0.3:
                risk = "🟢 LOW"
            elif phishing_prob < 0.7:
                risk = "🟡 MEDIUM"
            else:
                risk = "🔴 HIGH"
            
            # Check if prediction matches expected
            status = "✓" if pred.lower() == expected.lower() else "✗"
            
            print(f"{url[:48]:<50} {expected:<12} {pred:<12} {phishing_prob:.3f}{' ':<8} {risk:<10} {status}")
            
        except Exception as e:
            print(f"{url[:48]:<50} ERROR: {str(e)}")
    
    print("-"*90)

def save_model_for_flask(pipeline, best_model_name, final_metrics, feature_extractor):
    """Save model and metadata for Flask integration"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the complete pipeline
    pipeline_file = 'URLShield/models/phishing_detector.pkl'
    pipeline.save(pipeline_file)
    
    # Create metadata for Flask app
    metadata = {
        'model_name': best_model_name,
        'model_version': '1.0.0',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feature_names': feature_extractor.feature_names_,
        'performance_metrics': final_metrics,
        'test_data_stats': {
            'accuracy': final_metrics['accuracy'],
            'f1_score': final_metrics['f1_score'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'auc': final_metrics['auc']
        },
        'api_ready': True,
        'example_usage': {
            'python': "pipeline.predict(['https://example.com'])",
            'api_endpoint': "/predict",
            'input_format': "{'urls': ['url1', 'url2']}",
            'output_format': "{'predictions': ['legitimate/phishing'], 'probabilities': [[p_legit, p_phish]]}"
        }
    }
    
    joblib.dump(metadata, 'URLShield/models/model_metadata.pkl')
    
    # Save a simple JSON for web team
    import json
    web_metadata = {
        'model': best_model_name,
        'accuracy': float(final_metrics['accuracy']),
        'f1_score': float(final_metrics['f1_score']),
        'last_trained': metadata['training_date'],
        'total_features': len(feature_extractor.feature_names_),
        'api_endpoint': '/api/predict'
    }
    
    with open('URLShield/models/model_info.json', 'w') as f:
        json.dump(web_metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("💾 MODEL ARTIFACTS SAVED")
    print("="*70)
    print(f"📦 Complete pipeline: {pipeline_file}")
    print(f"📄 Model metadata: models/model_metadata.pkl")
    print(f"🌐 Web team JSON: models/model_info.json")
    
    # Test loading
    print("\n🧪 Testing model loading...")
    try:
        loaded_pipeline = UnifiedPhishingPipeline.load(pipeline_file)
        test_result = loaded_pipeline.predict(["https://www.google.com"])
        print(f"✅ Model loaded successfully! Test prediction: {test_result[0]}")
        
        # Show API usage example
        print("\n🔧 Flask API Usage Example:")
        print("""
        from flask import Flask, request, jsonify
        import joblib
        
        app = Flask(__name__)
        model = joblib.load('URLShield/models/phishing_detector.pkl')
        
        @app.route('/api/predict', methods=['POST'])
        def predict():
            data = request.json
            urls = data.get('urls', [])
            predictions = model.predict(urls)
            probabilities = model.predict_proba(urls).tolist()
            return jsonify({
                'predictions': predictions,
                'probabilities': probabilities
            })
        """)
        
    except Exception as e:
        print(f"❌ Error loading saved model: {e}")

def plot_model_comparison(results):
    """Create visualization of model comparison from sample evaluation"""
    # Filter out failed models
    valid_results = {k: v for k, v in results.items() if v['model'] is not None}
    
    if not valid_results:
        print("No valid models to plot!")
        return
    
    models = list(valid_results.keys())
    
    # Sort by F1-score
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    sorted_model_names = [m[0] for m in sorted_models]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    for idx, metric in enumerate(metrics):
        # Get values in sorted order
        values = [valid_results[model][metric] for model in sorted_model_names]
        
        bars = axes[idx].bar(range(len(sorted_model_names)), values, color=colors)
        
        # Add value labels
        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison', fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_xticks(range(len(sorted_model_names)))
        axes[idx].set_xticklabels(sorted_model_names, rotation=45, ha='right')
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Performance on Sample Data (Sorted by F1-Score)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('URLShield/models/model_comparison_sample.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model comparison plot saved: models/model_comparison_sample.png")

def main():
    """Main training workflow"""
    print("="*70)
    print("🚀 PHISHING DETECTION MODEL TRAINING PIPELINE")
    print("="*70)
    print("📋 Workflow:")
    print("   1. Load sample data for model selection")
    print("   2. Train & evaluate multiple models on sample")
    print("   3. Select best performing model")
    print("   4. Retrain best model on full dataset")
    print("   5. Save final model for Flask integration")
    print("="*70)
    
    # Step 1: Prepare sample data for model selection
    X_train_sample, X_val_sample, y_train_sample, y_val_sample, df = prepare_sample_data(sample_size=50000)
    
    # Step 2: Extract features from sample data
    X_train_sample_features, X_val_sample_features, _, feature_extractor = extract_features(
        X_train_sample, X_val_sample
    )
    
    # Convert sample labels to binary
    y_train_sample_binary = (y_train_sample == 'phishing').astype(int)
    y_val_sample_binary = (y_val_sample == 'phishing').astype(int)
    
    # Step 3: Select best model using sample data
    best_model, best_model_name, sample_results = select_best_model_on_sample(
        X_train_sample_features, y_train_sample_binary,
        X_val_sample_features, y_val_sample_binary
    )
    
    # Create visualization of sample results
    plot_model_comparison(sample_results)
    
    # Step 4: Prepare full data for final training
    X_train_full, X_test_full, y_train_full, y_test_full = prepare_full_data(df)
    
    # Extract features from full data
    X_train_full_features, X_test_full_features, _, _ = extract_features(
        X_train_full, X_test_full
    )
    
    # Convert full labels to binary
    y_train_full_binary = (y_train_full == 'phishing').astype(int)
    y_test_full_binary = (y_test_full == 'phishing').astype(int)
    
    # Step 5: Train final model on full data
    final_pipeline, final_metrics = train_final_model_on_full_data(
    best_model, best_model_name, feature_extractor,
    X_train_full, y_train_full_binary,  # Pass the raw URLs, not features
    X_test_full, y_test_full_binary
    )
    
    # Step 6: Test with single URLs
    test_single_predictions(final_pipeline)
    
    # Step 7: Save model for Flask integration
    save_model_for_flask(final_pipeline, best_model_name, final_metrics, feature_extractor)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"🏆 Selected Model: {best_model_name}")
    print(f"📊 Final Test Performance:")
    print(f"   Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {final_metrics['f1_score']:.4f}")
    print(f"   Precision: {final_metrics['precision']:.4f}")
    print(f"   Recall:    {final_metrics['recall']:.4f}")
    print(f"📁 Model saved: URLShield/models/phishing_detector.pkl")
    print(f"🔌 Ready for Flask integration!")
    print("="*70)

    # Test loading and using the saved model
    print("\nTesting saved model load...")
    try:
        loaded_pipeline = UnifiedPhishingPipeline.load('URLShield/models/phishing_detector.pkl')
        test_result = loaded_pipeline.predict(["https://www.google.com"])
        print(f"✓ Model loaded successfully. Test prediction: {test_result[0]}")
        
    except Exception as e:
        print(f"✗ Error loading saved model: {e}")
        import traceback
        traceback.print_exc()  # This will show the full error trace
    

if __name__ == "__main__":
    main()