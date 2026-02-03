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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor that works within sklearn pipeline"""
    
    def __init__(self):
        self.phishing_keywords = [
            'login', 'signin', 'verify', 'secure', 'account', 
            'update', 'confirm', 'banking', 'password', 'paypal',
            'alert', 'urgent', 'suspicion', 'limited', 'verification',
            'bank', 'security', 'update', 'click', 'login',
            'password', 'verify', 'account', 'secure', 'ebayisapi'
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

class ModelSelector:
    """Handles model training, evaluation, and selection"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def define_models(self):
        """Define all candidate models without imbalance handling"""
        models = {
            # Traditional Models
            'logistic_regression': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ))
            ]),
            
            'svm': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    kernel='rbf',
                    random_state=42,
                    probability=True
                ))
            ]),
            
            'knn': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', KNeighborsClassifier(
                    n_neighbors=5
                ))
            ]),
            
            'decision_tree': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', DecisionTreeClassifier(
                    random_state=42,
                    max_depth=20,
                    min_samples_split=10
                ))
            ]),
            
            # Ensemble Models
            'random_forest': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            
            'xgboost': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', XGBClassifier(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=-1
                ))
            ]),
            
            'catboost': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', CatBoostClassifier(
                    iterations=100,
                    depth=8,
                    learning_rate=0.1,
                    verbose=False,
                    random_state=42
                ))
            ]),
            
            'gradient_boosting': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                ))
            ]),
            
            'adaboost': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', AdaBoostClassifier(
                    n_estimators=100,
                    random_state=42
                ))
            ]),
            
            # Ensemble of ensembles
            'voting_ensemble': Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                        ('xgb', XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False)),
                        ('cb', CatBoostClassifier(iterations=50, verbose=False, random_state=42))
                    ],
                    voting='soft'
                ))
            ])
        }
        return models
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        """Train and evaluate all models on sample data"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        self.models = self.define_models()
        
        print("\n" + "="*70)
        print("TRAINING AND EVALUATING MODELS ON SAMPLE DATA")
        print("="*70)
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                results = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'model': model
                }
                
                # Add AUC if available
                if y_pred_proba is not None:
                    try:
                        auc = roc_auc_score(y_val, y_pred_proba)
                        results['auc'] = auc
                    except:
                        results['auc'] = 0.0
                
                self.results[name] = results
                
                print(f"  ✓ Accuracy: {accuracy:.4f}")
                print(f"  ✓ Precision: {precision:.4f}")
                print(f"  ✓ Recall: {recall:.4f}")
                print(f"  ✓ F1-Score: {f1:.4f}")
                if 'auc' in results:
                    print(f"  ✓ AUC: {results['auc']:.4f}")
                    
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)}")
                # Create dummy results for failed model
                self.results[name] = {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0,
                    'auc': 0,
                    'model': None,
                    'error': str(e)
                }
        
        # Select best model based on F1-score
        self.select_best_model()
    
    def select_best_model(self):
        """Select the best model based on F1-score"""
        best_f1 = -1
        best_name = None
        
        for name, result in self.results.items():
            if result['model'] is not None and result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                best_name = name
        
        if best_name is None:
            # Fallback to accuracy if all models failed
            for name, result in self.results.items():
                if result['model'] is not None and result['accuracy'] > best_f1:
                    best_f1 = result['accuracy']
                    best_name = name
        
        if best_name is not None:
            self.best_model_name = best_name
            self.best_model = self.results[best_name]['model']
            
            print("\n" + "="*70)
            print(f"🏆 BEST MODEL SELECTED: {best_name}")
            print(f"Best F1-Score on validation: {best_f1:.4f}")
            print("="*70)
        else:
            print("\n❌ Warning: No valid model found!")
            self.best_model_name = None
            self.best_model = None
    
    def get_best_model_info(self):
        """Return the best model and its name"""
        return self.best_model, self.best_model_name
    
    def get_all_results(self):
        """Return all evaluation results"""
        return self.results
    
    def print_model_comparison(self):
        """Print comparison of all models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY (Sample Data)")
        print("="*70)
        
        headers = ["Rank", "Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Status"]
        rows = []
        
        valid_results = {k: v for k, v in self.results.items() if v['model'] is not None}
        
        # Sort by F1-score
        sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for rank, (name, result) in enumerate(sorted_models, 1):
            row = [
                rank,
                name,
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1_score']:.4f}",
                f"{result.get('auc', 0):.4f}",
                "✓"
            ]
            rows.append(row)
        
        # Add failed models
        failed_models = {k: v for k, v in self.results.items() if v['model'] is None}
        for name, result in failed_models.items():
            row = [
                "-",
                name,
                "0.0000",
                "0.0000",
                "0.0000",
                "0.0000",
                "0.0000",
                "✗"
            ]
            rows.append(row)
        
        # Print table
        col_widths = [max(len(str(item)) for item in col) for col in zip(*[headers] + rows)]
        
        # Print headers
        header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_row)
        print("-" * len(header_row))
        
        # Print rows
        for row in rows:
            print(" | ".join(str(item).ljust(w) for item, w in zip(row, col_widths)))

class UnifiedPhishingPipeline:
    """Complete pipeline from URL to prediction in one object"""
    
    def __init__(self, model=None, feature_extractor=None):
        if feature_extractor is None:
            self.feature_extractor = URLFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor
        
        if model is None:
            # Create a default pipeline
            from sklearn.ensemble import RandomForestClassifier
            self.pipeline = Pipeline([
                ('feature_extractor', self.feature_extractor),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42
                ))
            ])
        else:
            # Wrap the provided model in a complete pipeline
            self.pipeline = Pipeline([
                ('feature_extractor', self.feature_extractor),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
    
    @classmethod
    def from_pipeline(cls, pipeline):
        """Create UnifiedPhishingPipeline from an existing sklearn pipeline"""
        instance = cls()
        instance.pipeline = pipeline
        instance.feature_extractor = pipeline.named_steps['feature_extractor']
        return instance
    
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
        # ✅ FIX: Save the sklearn pipeline directly
        joblib.dump(self.pipeline, filepath)
        print(f"✓ Complete pipeline saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load a saved pipeline"""
        loaded_pipeline = joblib.load(filepath)
        instance = UnifiedPhishingPipeline()
        instance.pipeline = loaded_pipeline
        instance.feature_extractor = loaded_pipeline.named_steps['feature_extractor']
        return instance