import os
import re
import math
import ipaddress
import pandas as pd
from urllib.parse import urlparse


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "PhiUSIIL_Phishing_URL_Dataset.csv"
)


# ============================================================
# THE 30 FEATURES USED BY THE HYBRID ENSEMBLE MODEL
# ============================================================

MODEL_FEATURES = [
    # --- Original 21 ---
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoofobfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOFQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "ISHTTPS",
    "Bank",
    "Pay",
    "Crypto",
    # --- NEW 9 ---
    "URLEntropy",          # Shannon entropy of full URL (high → random/obfuscated)
    "DomainEntropy",       # Shannon entropy of domain portion
    "IsPunycode",          # 1 if domain contains xn-- (homograph attack)
    "IsSuspiciousTLD",     # 1 if TLD in known-phishing TLD list
    "HasRedirectInURL",    # 1 if URL contains "http://" or "https://" more than once
    "URLDepth",            # Number of path segments (slashes in path)
    "SubdomainLength",     # Total character length of all subdomain labels
    "HasBrandMismatch",    # 1 if well-known brand keyword in path but NOT in domain
    "DomainHyphenCount",   # Number of hyphens in domain (common in phishing)
]


# ============================================================
# SUSPICIOUS TLD SET
# ============================================================

SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "top", "click", "loan", "gq",
    "pw", "cc", "su", "ws", "biz", "info", "mobi", "name", "online",
    "site", "space", "tech", "live", "store", "shop", "icu", "buzz",
    "work", "fun", "uno", "cyou", "rest", "world", "monster"
}

# Brands whose name appearing in the URL path (but not domain) signals impersonation
BRAND_KEYWORDS = [
    "paypal", "apple", "amazon", "google", "facebook", "instagram",
    "netflix", "microsoft", "yahoo", "ebay", "twitter", "linkedin",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hdfc", "sbi",
    "icici", "axisbank", "steam", "roblox", "coinbase", "binance",
]


# ============================================================
# HELPER: SHANNON ENTROPY
# ============================================================

def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ============================================================
# LOAD DATASET (FOR SEPARATE TRAINING CODE)
# ============================================================

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    return df


# ============================================================
# CHECK WHETHER DOMAIN IS AN IP ADDRESS
# ============================================================

def is_domain_ip(domain):
    try:
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        return 0


# ============================================================
# EXTRACT 30 FEATURES FROM A SINGLE URL
# ============================================================

def extract_url_features(url):
    url_clean = str(url).strip()
    if not url_clean:
        return {feat: 0 for feat in MODEL_FEATURES}

    # Ensure URL has a scheme for proper parsing
    parsed = urlparse(url_clean)
    if not parsed.netloc:
        parsed = urlparse("http://" + url_clean)

    domain = parsed.netloc.lower()

    # Remove username/password and port
    if "@" in domain:
        domain = domain.split("@")[-1]
    if ":" in domain:
        domain = domain.split(":")[0]

    # Domain core without www. for accurate lexical analysis
    dom_core = domain[4:] if domain.startswith("www.") else domain
    parts = domain.split(".") if domain else []
    tld = parts[-1] if len(parts) >= 2 else ""
    tld_length = len(tld)

    # Core path and query
    path_query = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")
    core_str = dom_core + path_query

    # ---- Original 21 features ----

    url_length = len(url_clean)
    domain_length = len(domain)
    domain_is_ip = is_domain_ip(domain)

    # Subdomains (excluding www and the root domain label)
    no_of_subdomain = max(len(parts) - 2, 0)

    # Obfuscation
    obfuscated_characters = url_clean.count("%") + url_clean.count("@")
    has_obfuscation = 1 if obfuscated_characters > 0 else 0
    no_of_obfuscated_char = obfuscated_characters
    obfuscation_ratio = (obfuscated_characters / url_length) if url_length > 0 else 0.0

    # Letters
    domain_body = dom_core[:-len(tld)-1] if len(parts) >= 2 and len(dom_core) > len(tld) + 1 else dom_core
    no_of_letters = (
        sum(1 for char in domain_body if char.isalpha())
        + max(tld_length - 1, 0)
        + sum(1 for char in path_query if char.isalpha())
    )
    letter_ratio = (no_of_letters / url_length) if url_length > 0 else 0.0

    # Digits
    no_of_digits = sum(1 for char in core_str if char.isdigit())
    digit_ratio = (no_of_digits / url_length) if url_length > 0 else 0.0

    # Special characters
    no_of_equals = url_clean.count("=")
    no_of_question_marks = url_clean.count("?")
    no_of_ampersand = url_clean.count("&")
    special_characters = set("!#$%&'()*+,-/:;=?@[]^_`{|}~.")
    no_of_other_special_chars = sum(
        1 for char in core_str
        if char in special_characters and char not in ["=", "?", "&"]
    )
    total_special = no_of_equals + no_of_question_marks + no_of_ampersand + no_of_other_special_chars
    special_char_ratio = (total_special / url_length) if url_length > 0 else 0.0

    # HTTPS check
    is_https = 1 if parsed.scheme.lower() == "https" else 0

    # Suspicious keywords in URL
    url_lower = url_clean.lower()
    bank_keywords = ["bank", "banking", "account", "netbanking", "creditcard", "debitcard"]
    pay_keywords = ["pay", "payment", "checkout", "billing", "invoice", "paypal"]
    crypto_keywords = ["crypto", "bitcoin", "ethereum", "wallet", "usdt", "binance"]

    bank = int(any(k in url_lower for k in bank_keywords))
    pay = int(any(k in url_lower for k in pay_keywords))
    crypto = int(any(k in url_lower for k in crypto_keywords))

    # ---- New 9 features ----

    # Shannon entropy
    url_entropy = round(_shannon_entropy(url_clean), 6)
    domain_entropy = round(_shannon_entropy(dom_core), 6)

    # Punycode / homograph attack detection
    is_punycode = 1 if "xn--" in domain.lower() else 0

    # Suspicious TLD
    is_suspicious_tld = 1 if tld.lower() in SUSPICIOUS_TLDS else 0

    # Redirect-in-URL: second occurrence of http
    http_count = url_lower.count("http://") + url_lower.count("https://")
    has_redirect_in_url = 1 if http_count > 1 else 0

    # URL depth (number of path segments)
    path_segments = [s for s in parsed.path.split("/") if s]
    url_depth = len(path_segments)

    # Subdomain total character length (labels excluding SLD and TLD)
    subdomain_labels = parts[:-2] if len(parts) > 2 else []
    subdomain_length = sum(len(lbl) for lbl in subdomain_labels)

    # Brand mismatch: brand name in path/query but NOT in domain
    domain_lower = dom_core.lower()
    path_lower = path_query.lower()
    has_brand_mismatch = 0
    for brand in BRAND_KEYWORDS:
        if brand in path_lower and brand not in domain_lower:
            has_brand_mismatch = 1
            break

    # Hyphen count in domain (e.g. paypal-secure-login.xyz)
    domain_hyphen_count = dom_core.count("-")

    features = {
        # Original 21
        "URLLength": int(url_length),
        "DomainLength": int(domain_length),
        "IsDomainIP": int(domain_is_ip),
        "TLDLength": int(tld_length),
        "NoOfSubDomain": int(no_of_subdomain),
        "HasObfuscation": int(has_obfuscation),
        "NoofobfuscatedChar": int(no_of_obfuscated_char),
        "ObfuscationRatio": round(float(obfuscation_ratio), 6),
        "NoOfLettersInURL": int(no_of_letters),
        "LetterRatioInURL": round(float(letter_ratio), 6),
        "NoOfDegitsInURL": int(no_of_digits),
        "DegitRatioInURL": round(float(digit_ratio), 6),
        "NoOfEqualsInURL": int(no_of_equals),
        "NoOFQMarkInURL": int(no_of_question_marks),
        "NoOfAmpersandInURL": int(no_of_ampersand),
        "NoOfOtherSpecialCharsInURL": int(no_of_other_special_chars),
        "SpacialCharRatioInURL": round(float(special_char_ratio), 6),
        "ISHTTPS": int(is_https),
        "Bank": int(bank),
        "Pay": int(pay),
        "Crypto": int(crypto),
        # New 9
        "URLEntropy": float(url_entropy),
        "DomainEntropy": float(domain_entropy),
        "IsPunycode": int(is_punycode),
        "IsSuspiciousTLD": int(is_suspicious_tld),
        "HasRedirectInURL": int(has_redirect_in_url),
        "URLDepth": int(url_depth),
        "SubdomainLength": int(subdomain_length),
        "HasBrandMismatch": int(has_brand_mismatch),
        "DomainHyphenCount": int(domain_hyphen_count),
    }

    return features


# ============================================================
# PREPARE DATA (FOR PREDICTION OR TRAINING)
# ============================================================

def prepare_data(df):
    """
    Prepares feature DataFrame X (and optional target y).
    Guarantees no 'label not found' error occurs during URL prediction.
    """
    if "URL" in df.columns:
        features_list = [extract_url_features(u) for u in df["URL"]]
        X = pd.DataFrame(features_list, columns=MODEL_FEATURES)
        y = df["label"] if "label" in df.columns else None
        return X, y

    # For dataset DataFrames (training flow)
    missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing required features: {missing_features}")

    X = df[MODEL_FEATURES].copy()
    y = df["label"] if "label" in df.columns else None
    return X, y


# ============================================================
# TEST FEATURE EXTRACTION
# ============================================================

if __name__ == "__main__":
    print("\n========================================")
    print("       URL FEATURE EXTRACTOR (v2)")
    print("========================================")

    test_url = input("\nEnter a URL to test: ").strip() or "https://www.google.com"
    features = extract_url_features(test_url)

    print(f"\nExtracted Features ({len(MODEL_FEATURES)} total):")
    for name in MODEL_FEATURES:
        print(f"  {name:30}: {features[name]}")

    test_df = pd.DataFrame({"URL": [test_url]})
    X, y = prepare_data(test_df)
    print("\nFeature matrix shape:", X.shape)
    print("Feature extraction completed successfully!")