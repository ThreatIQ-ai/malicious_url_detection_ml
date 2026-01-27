import re
import tldextract
import numpy as np
import pandas as pd
from urllib.parse import urlparse, parse_qs
from sklearn.base import BaseEstimator, TransformerMixin

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extractor for URL phishing detection
    Extracts 45+ features from URLs
    """
    
    def __init__(self):
        self.top_domains = None
        self.phishing_keywords = [
            'login', 'signin', 'verify', 'secure', 'account', 
            'update', 'confirm', 'banking', 'password', 'paypal',
            'alert', 'urgent', 'suspension', 'limited', 'verification'
        ]
        
    def fit(self, X, y=None):
        # Learn top legitimate domains from training data
        if y is not None:
            legit_urls = X[y == 'legitimate']
            self.top_domains = self._extract_top_domains(legit_urls)
        return self
    
    def transform(self, X, y=None):
        features = []
        
        for url in X:
            url = str(url).strip().lower()
            feature_dict = {}
            
            # 1. Basic URL features
            feature_dict.update(self._extract_basic_features(url))
            
            # 2. Domain features
            feature_dict.update(self._extract_domain_features(url))
            
            # 3. Path features
            feature_dict.update(self._extract_path_features(url))
            
            # 4. Query features
            feature_dict.update(self._extract_query_features(url))
            
            # 5. Special character features
            feature_dict.update(self._extract_special_char_features(url))
            
            # 6. Suspicious patterns
            feature_dict.update(self._extract_suspicious_patterns(url))
            
            # 7. Lexical features
            feature_dict.update(self._extract_lexical_features(url))
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _extract_basic_features(self, url):
        features = {}
        
        # URL length features
        features['url_length'] = len(url)
        features['log_url_length'] = np.log1p(len(url))
        
        # Protocol
        features['has_https'] = 1 if url.startswith('https') else 0
        features['has_http'] = 1 if url.startswith('http') else 0
        
        # www presence
        features['has_www'] = 1 if 'www.' in url else 0
        
        return features
    
    def _extract_domain_features(self, url):
        features = {}
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
            ext = tldextract.extract(url)
            
            # Domain length
            features['domain_length'] = len(ext.domain)
            features['subdomain_length'] = len('.'.join(filter(None, [ext.subdomain, ext.domain])))
            
            # Subdomain features
            subdomains = ext.subdomain.split('.') if ext.subdomain else []
            features['num_subdomains'] = len(subdomains)
            features['has_subdomain'] = 1 if ext.subdomain else 0
            
            # TLD features
            features['tld_length'] = len(ext.suffix)
            features['is_common_tld'] = 1 if ext.suffix in ['.com', '.org', '.net', '.edu', '.gov'] else 0
            features['is_uncommon_tld'] = 1 if ext.suffix in ['.xyz', '.tk', '.ml', '.ga', '.cf', '.gq'] else 0
            
            # Domain entropy (randomness measure)
            features['domain_entropy'] = self._calculate_entropy(ext.domain)
            
            # Digits in domain
            features['domain_digit_ratio'] = sum(c.isdigit() for c in ext.domain) / max(1, len(ext.domain))
            
            # Hyphens in domain
            features['domain_hyphen_count'] = ext.domain.count('-')
            
        except:
            # Default values if parsing fails
            features.update({
                'domain_length': 0,
                'subdomain_length': 0,
                'num_subdomains': 0,
                'has_subdomain': 0,
                'tld_length': 0,
                'is_common_tld': 0,
                'is_uncommon_tld': 0,
                'domain_entropy': 0,
                'domain_digit_ratio': 0,
                'domain_hyphen_count': 0
            })
        
        return features
    
    def _extract_path_features(self, url):
        features = {}
        
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            features['path_length'] = len(path)
            features['path_depth'] = path.count('/')
            features['has_file_extension'] = 1 if '.' in path.split('/')[-1] and len(path.split('/')[-1].split('.')) > 1 else 0
            
            # Common extensions
            if '.' in path:
                ext = path.split('.')[-1].lower()
                features['is_php'] = 1 if ext == 'php' else 0
                features['is_html'] = 1 if ext in ['html', 'htm'] else 0
                features['is_asp'] = 1 if ext == 'asp' else 0
                features['is_exe'] = 1 if ext == 'exe' else 0
                features['is_zip'] = 1 if ext == 'zip' else 0
            
        except:
            features.update({
                'path_length': 0,
                'path_depth': 0,
                'has_file_extension': 0,
                'is_php': 0,
                'is_html': 0,
                'is_asp': 0,
                'is_exe': 0,
                'is_zip': 0
            })
        
        return features
    
    def _extract_query_features(self, url):
        features = {}
        
        try:
            parsed = urlparse(url)
            query = parsed.query
            
            features['has_query'] = 1 if query else 0
            features['query_length'] = len(query)
            features['num_params'] = len(parse_qs(query))
            
            # Suspicious parameters
            suspicious_params = ['login', 'password', 'user', 'account', 'verify', 'token', 'auth']
            features['has_suspicious_param'] = 0
            
            if query:
                params = parse_qs(query)
                for param in params:
                    if any(keyword in param.lower() for keyword in suspicious_params):
                        features['has_suspicious_param'] = 1
                        break
            
        except:
            features.update({
                'has_query': 0,
                'query_length': 0,
                'num_params': 0,
                'has_suspicious_param': 0
            })
        
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
        
        # Ratios
        features['special_char_ratio'] = sum([
            features['count_dash'], features['count_underscore'],
            features['count_dot'], features['count_question'],
            features['count_equal'], features['count_ampersand'],
            features['count_percent'], features['count_slash'],
            features['count_at']
        ]) / max(1, len(url))
        
        # IP address detection
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        features['has_ip_address'] = 1 if re.search(ip_pattern, url) else 0
        
        return features
    
    def _extract_suspicious_patterns(self, url):
        features = {}
        
        # Shortening services
        shortening_services = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly', 'qrco.de']
        features['is_shortened'] = 1 if any(service in url for service in shortening_services) else 0
        
        # Phishing keywords in URL
        features['phishing_keyword_count'] = sum(1 for keyword in self.phishing_keywords if keyword in url)
        
        # Hex characters
        features['hex_ratio'] = len(re.findall(r'[0-9a-fA-F]{4,}', url)) / max(1, len(url))
        
        # Double slashes
        features['has_double_slash'] = 1 if '//' in url[7:] else 0  # Skip http://
        
        # Port number
        features['has_port'] = 1 if re.search(r':\d{2,5}', url) else 0
        
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
        
        # Digit ratio
        features['digit_ratio'] = sum(c.isdigit() for c in url) / max(1, len(url))
        
        # Letter ratio
        features['letter_ratio'] = sum(c.isalpha() for c in url) / max(1, len(url))
        
        return features
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of a string"""
        if not text:
            return 0
        entropy = 0
        for char in set(text):
            p = text.count(char) / len(text)
            entropy -= p * np.log2(p)
        return entropy
    
    def _extract_top_domains(self, urls, n=1000):
        """Extract top domains from URLs"""
        domains = []
        for url in urls:
            try:
                ext = tldextract.extract(url)
                domains.append(f"{ext.domain}.{ext.suffix}")
            except:
                continue
        
        from collections import Counter
        return Counter(domains).most_common(n)