import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import re
import tldextract
from urllib.parse import urlparse, parse_qs

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor that works within sklearn pipeline"""
    
    def __init__(self):
        self.phishing_keywords = [
            'login', 'signin', 'verify', 'secure', 'account', 
            'update', 'confirm', 'banking', 'password', 'paypal',
            'alert', 'urgent', 'suspension', 'limited', 'verification'
        ]
    
    def fit(self, X, y=None):
        return self  # Nothing to fit during feature extraction
    
    def transform(self, X, y=None):
        """Transform URLs into features"""
        features = []
        for url in X:
            url = str(url).strip().lower()
            feature_dict = {}
            
            # Extract all features
            feature_dict.update(self._extract_basic_features(url))
            feature_dict.update(self._extract_domain_features(url))
            feature_dict.update(self._extract_path_features(url))
            feature_dict.update(self._extract_query_features(url))
            feature_dict.update(self._extract_special_char_features(url))
            feature_dict.update(self._extract_suspicious_patterns(url))
            feature_dict.update(self._extract_lexical_features(url))
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    # All the feature extraction methods (same as before)
    def _extract_basic_features(self, url):
        features = {}
        features['url_length'] = len(url)
        features['has_https'] = 1 if url.startswith('https') else 0
        features['has_http'] = 1 if url.startswith('http') else 0
        features['has_www'] = 1 if 'www.' in url else 0
        return features
    
    def _extract_domain_features(self, url):
        features = {}
        try:
            ext = tldextract.extract(url)
            features['domain_length'] = len(ext.domain)
            features['num_subdomains'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
            features['has_subdomain'] = 1 if ext.subdomain else 0
            features['tld_length'] = len(ext.suffix)
            features['is_common_tld'] = 1 if ext.suffix in ['.com', '.org', '.net'] else 0
            features['domain_digit_ratio'] = sum(c.isdigit() for c in ext.domain) / max(1, len(ext.domain))
        except:
            features.update({k: 0 for k in ['domain_length', 'num_subdomains', 'has_subdomain', 
                                           'tld_length', 'is_common_tld', 'domain_digit_ratio']})
        return features
    
    # ... (include all other feature extraction methods from the original)
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy"""
        if not text:
            return 0
        entropy = 0
        for char in set(text):
            p = text.count(char) / len(text)
            entropy -= p * np.log2(p)
        return entropy

class UnifiedPhishingPipeline:
    """Complete pipeline from URL to prediction in one object"""
    
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.pipeline = self._create_unified_pipeline()
    
    def _create_unified_pipeline(self):
        """Create a single pipeline with feature extraction + model"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier
        
        # Create the complete pipeline
        pipeline = Pipeline([
            ('feature_extractor', URLFeatureExtractor()),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', self._get_model())
        ])
        
        return pipeline
    
    def _get_model(self):
        """Get the appropriate model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
        elif self.model_type == 'catboost':
            return CatBoostClassifier(
                iterations=200,
                depth=10,
                learning_rate=0.1,
                verbose=False,
                random_state=42
            )
        else:  # ensemble
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            )
            xgb = XGBClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                use_label_encoder=False
            )
            cb = CatBoostClassifier(
                iterations=100,
                depth=8,
                learning_rate=0.1,
                verbose=False,
                random_state=42
            )
            return VotingClassifier(
                estimators=[('rf', rf), ('xgb', xgb), ('cb', cb)],
                voting='soft'
            )
    
    def fit(self, X, y):
        """Train the complete pipeline"""
        # Convert labels to binary
        y_binary = (np.array(y) == 'phishing').astype(int)
        self.pipeline.fit(X, y_binary)
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.pipeline.predict(X)
        return ['phishing' if pred == 1 else 'legitimate' for pred in predictions]
    
    def predict_proba(self, X):
        """Get probability predictions"""
        return self.pipeline.predict_proba(X)
    
    def save(self, filepath):
        """Save the complete pipeline to a single .pkl file"""
        joblib.dump(self.pipeline, filepath)
        print(f"Complete pipeline saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load a saved pipeline"""
        pipeline = joblib.load(filepath)
        return pipeline