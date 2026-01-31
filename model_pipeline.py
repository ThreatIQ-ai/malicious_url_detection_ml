from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import re
from xgboost import XGBClassifier
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
        # Store all feature names to ensure consistency
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        # Transform once to get feature names
        X_transformed = self.transform(X)
        self.feature_names_ = list(X_transformed.columns)
        return self
    
    def transform(self, X, y=None):
        """Transform URLs into features - always returns consistent features"""
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
        
        # Convert to DataFrame with consistent columns
        result_df = pd.DataFrame(features)
        
        # Ensure all expected features exist (fill missing with 0)
        expected_features = self._get_all_feature_names()
        for feature in expected_features:
            if feature not in result_df.columns:
                result_df[feature] = 0
        
        # Return features in consistent order
        return result_df[expected_features]
    
    def _get_all_feature_names(self):
        """Return list of all possible feature names"""
        return [
            # Basic features
            'url_length', 'has_https', 'has_http', 'has_www',
            
            # Domain features
            'domain_length', 'num_subdomains', 'has_subdomain',
            'tld_length', 'is_common_tld', 'domain_digit_ratio',
            'domain_hyphen_count', 'domain_entropy',
            
            # Path features
            'path_length', 'path_depth', 'has_file_extension',
            'is_php', 'is_html', 'is_asp', 'is_exe', 'is_zip',
            
            # Query features
            'has_query', 'query_length', 'num_params', 'has_suspicious_param',
            
            # Special characters
            'count_dash', 'count_underscore', 'count_dot', 'count_question',
            'count_equal', 'count_ampersand', 'count_percent', 'count_slash',
            'count_at', 'special_char_ratio', 'has_ip_address',
            
            # Suspicious patterns
            'is_shortened', 'phishing_keyword_count', 'hex_ratio',
            'has_double_slash', 'has_port',
            
            # Lexical features
            'vowel_ratio', 'consonant_ratio', 'digit_ratio', 'letter_ratio'
        ]
    
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
            
            subdomains = ext.subdomain.split('.') if ext.subdomain else []
            features['num_subdomains'] = len(subdomains)
            features['has_subdomain'] = 1 if ext.subdomain else 0
            
            features['tld_length'] = len(ext.suffix)
            features['is_common_tld'] = 1 if ext.suffix in ['.com', '.org', '.net', '.edu', '.gov'] else 0
            
            # Calculate digit ratio
            domain = ext.domain
            features['domain_digit_ratio'] = sum(c.isdigit() for c in domain) / max(1, len(domain))
            features['domain_hyphen_count'] = domain.count('-')
            features['domain_entropy'] = self._calculate_entropy(domain)
            
        except:
            # Set default values
            features.update({
                'domain_length': 0,
                'num_subdomains': 0,
                'has_subdomain': 0,
                'tld_length': 0,
                'is_common_tld': 0,
                'domain_digit_ratio': 0,
                'domain_hyphen_count': 0,
                'domain_entropy': 0
            })
        
        return features
    
    def _extract_path_features(self, url):
        features = {
            'path_length': 0,
            'path_depth': 0,
            'has_file_extension': 0,
            'is_php': 0, 'is_html': 0, 'is_asp': 0, 'is_exe': 0, 'is_zip': 0
        }
        
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            features['path_length'] = len(path)
            features['path_depth'] = path.count('/')
            
            # Check for file extension
            if '.' in path:
                last_part = path.split('/')[-1]
                if '.' in last_part and len(last_part.split('.')) > 1:
                    features['has_file_extension'] = 1
                    ext = last_part.split('.')[-1].lower()
                    
                    # Check specific extensions
                    features['is_php'] = 1 if ext == 'php' else 0
                    features['is_html'] = 1 if ext in ['html', 'htm'] else 0
                    features['is_asp'] = 1 if ext == 'asp' else 0
                    features['is_exe'] = 1 if ext == 'exe' else 0
                    features['is_zip'] = 1 if ext == 'zip' else 0
                    
        except:
            pass    
        
        return features
    
    def _extract_query_features(self, url):
        features = {
            'has_query': 0,
            'query_length': 0,
            'num_params': 0,
            'has_suspicious_param': 0
        }
        
        try:
            parsed = urlparse(url)
            query = parsed.query
            
            if query:
                features['has_query'] = 1
                features['query_length'] = len(query)
                features['num_params'] = len(parse_qs(query))
                
                # Check for suspicious parameters
                suspicious_params = ['login', 'password', 'user', 'account', 'verify', 'token', 'auth']
                params = parse_qs(query)
                for param in params:
                    if any(keyword in param.lower() for keyword in suspicious_params):
                        features['has_suspicious_param'] = 1
                        break
                        
        except:
            pass
        
        return features
    
    def _extract_special_char_features(self, url):
        features = {}
        
        # Count special characters
        features['count_dash'] = url.count('-')
        features['count_underscore'] = url.count('_')
        features['count_dot'] = url.count('.')
        features['count_question'] = url.count('?')
        features['count_equal'] = url.count('=')
        features['count_ampersand'] = url.count('&')
        features['count_percent'] = url.count('%')
        features['count_slash'] = url.count('/')
        features['count_at'] = url.count('@')
        
        # Calculate ratio
        special_count = sum([
            features['count_dash'], features['count_underscore'],
            features['count_dot'], features['count_question'],
            features['count_equal'], features['count_ampersand'],
            features['count_percent'], features['count_slash'],
            features['count_at']
        ])
        features['special_char_ratio'] = special_count / max(1, len(url))
        
        # IP address detection
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        features['has_ip_address'] = 1 if re.search(ip_pattern, url) else 0
        
        return features
    
    def _extract_suspicious_patterns(self, url):
        features = {}
        
        # Shortening services
        shortening_services = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly', 'qrco.de']
        features['is_shortened'] = 1 if any(service in url for service in shortening_services) else 0
        
        # Phishing keywords
        features['phishing_keyword_count'] = sum(1 for keyword in self.phishing_keywords if keyword in url)
        
        # Hex characters
        hex_matches = re.findall(r'[0-9a-fA-F]{4,}', url)
        features['hex_ratio'] = len(''.join(hex_matches)) / max(1, len(url))
        
        # Double slashes (after protocol)
        features['has_double_slash'] = 1 if '//' in url[7:] else 0
        
        # Port number
        features['has_port'] = 1 if re.search(r':\d{2,5}/', url) or re.search(r':\d{2,5}$', url) else 0
        
        return features
    
    def _extract_lexical_features(self, url):
        features = {}
        
        # Vowel/consonant ratio in domain
        try:
            ext = tldextract.extract(url)
            domain = ext.domain
            vowels = sum(1 for c in domain if c in 'aeiou')
            consonants = sum(1 for c in domain if c.isalpha() and c not in 'aeiou')
            features['vowel_ratio'] = vowels / max(1, len(domain))
            features['consonant_ratio'] = consonants / max(1, len(domain))
        except:
            features['vowel_ratio'] = 0
            features['consonant_ratio'] = 0
        
        # Overall digit and letter ratios
        features['digit_ratio'] = sum(c.isdigit() for c in url) / max(1, len(url))
        features['letter_ratio'] = sum(c.isalpha() for c in url) / max(1, len(url))
        
        return features
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of a string"""
        if not text:
            return 0
        
        # Calculate frequency
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        length = len(text)
        for count in freq.values():
            p = count / length
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
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier
        
        # Create the complete pipeline
        pipeline = Pipeline([
            ('feature_extractor', URLFeatureExtractor()),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Changed to constant
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