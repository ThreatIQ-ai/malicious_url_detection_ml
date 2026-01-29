from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the SINGLE .pkl file containing complete pipeline
print("Loading unified phishing detection pipeline...")
try:
    # This single file contains feature extractor + model
    pipeline = joblib.load('models/phishing_detector.pkl')
    print("✓ Complete pipeline loaded from single .pkl file")
    print(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    try:
        # Load dataset for statistics
        df = pd.read_csv('data/url_dataset.csv')
        
        stats = {
            'total_urls': len(df),
            'legitimate': len(df[df['type'] == 'legitimate']),
            'phishing': len(df[df['type'] == 'phishing']),
            'unique_domains': df['url'].apply(lambda x: x.split('/')[2] if '//' in x else x).nunique() if 'url' in df.columns else 0,
            'training_samples': int(len(df) * 0.8),
            'testing_samples': int(len(df) * 0.2),
            'legitimate_percentage': round(len(df[df['type'] == 'legitimate']) / len(df) * 100, 2),
            'phishing_percentage': round(len(df[df['type'] == 'phishing']) / len(df) * 100, 2),
            'now': datetime.now()
        }
        
        return render_template('dashboard.html', **stats)
        
    except Exception as e:
        return render_template('dashboard.html', 
                             error=f"Could not load dashboard data: {str(e)}",
                             total_urls=731495,
                             legitimate=480194,
                             phishing=251301,
                             unique_domains=394837,
                             training_samples=585196,
                             testing_samples=146299,
                             legitimate_percentage=65.65,
                             phishing_percentage=34.35,
                             now=datetime.now())

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return render_template('result.html',
                             prediction="Error",
                             probability=0,
                             url="",
                             error="Model not loaded. Please train the model first.")
    
    try:
        # Get URL from form
        url = request.form.get('url', '').strip()
        
        if not url:
            return render_template('result.html',
                                 prediction="Invalid",
                                 probability=0,
                                 url="",
                                 error="Please enter a URL")
        
        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Use the SINGLE pipeline for prediction
        # Pipeline automatically extracts features and makes prediction
        proba = pipeline.predict_proba([url])[0]
        prediction = pipeline.predict([url])[0]
        
        # Format results
        phishing_prob = proba[1]
        legitimate_prob = proba[0]
        
        if phishing_prob > 0.6:
            risk_level = "HIGH"
            color = "danger"
            display_pred = "PHISHING"
        elif phishing_prob > 0.4:
            risk_level = "MEDIUM"
            color = "warning"
            display_pred = "SUSPICIOUS"
        else:
            risk_level = "LOW"
            color = "success"
            display_pred = "LEGITIMATE"
        
        # Extract features for display (optional)
        feature_extractor = pipeline.named_steps['feature_extractor']
        features_df = feature_extractor.transform([url])
        
        # Select important features to show
        important_features = {
            'URL Length': int(features_df['url_length'].iloc[0]),
            'Has HTTPS': 'Yes' if features_df['has_https'].iloc[0] == 1 else 'No',
            'Subdomains': int(features_df['num_subdomains'].iloc[0]),
            'Domain Length': int(features_df['domain_length'].iloc[0]),
            'TLD Type': 'Common' if features_df['is_common_tld'].iloc[0] == 1 else 'Uncommon',
            'Has WWW': 'Yes' if features_df['has_www'].iloc[0] == 1 else 'No'
        }
        
        return render_template('result.html',
                             prediction=display_pred,
                             probability=max(phishing_prob, legitimate_prob),
                             url=url,
                             risk_level=risk_level,
                             color=color,
                             features=important_features,
                             phishing_prob=f"{phishing_prob*100:.1f}%",
                             legitimate_prob=f"{legitimate_prob*100:.1f}%")
    
    except Exception as e:
        return render_template('result.html',
                             prediction="Error",
                             probability=0,
                             url=url if 'url' in locals() else "",
                             error=f"Prediction error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint that uses the single .pkl file"""
    if pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Single call to pipeline
        proba = pipeline.predict_proba([url])[0]
        
        return jsonify({
            'url': url,
            'phishing_probability': float(proba[1]),
            'legitimate_probability': float(proba[0]),
            'prediction': 'phishing' if proba[1] > 0.5 else 'legitimate',
            'confidence': float(max(proba))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    if pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({'error': 'No URLs provided'}), 400
        
        results = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
            
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            try:
                proba = pipeline.predict_proba([url])[0]
                results.append({
                    'url': url,
                    'phishing_probability': float(proba[1]),
                    'legitimate_probability': float(proba[0]),
                    'prediction': 'phishing' if proba[1] > 0.5 else 'legitimate'
                })
            except:
                results.append({
                    'url': url,
                    'phishing_probability': None,
                    'prediction': 'error'
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if pipeline:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)