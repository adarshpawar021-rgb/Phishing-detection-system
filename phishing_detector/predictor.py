"""
predictor.py — Hybrid URL Phishing Detection Engine

Four-layer fusion architecture:
  Layer 1: Allowlist / Blocklist fast-pass (zero-latency on known domains)
  Layer 2: ML Ensemble (RF + GBM + LR soft voting, 30 features)
  Layer 3: Deep Lexical Heuristics (17 independent rule checks)
  Layer 4: Final Score Fusion with calibration and risk-tier mapping

Returns a comprehensive result dict including:
  - result          : "LEGITIMATE" | "SUSPICIOUS" | "PHISHING"
  - risk_level      : "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
  - confidence      : float 0–100
  - risk_factors    : list of human-readable reasons triggered
  - method_scores   : per-layer raw scores for transparency
"""

import os
import joblib
import pandas as pd
from urllib.parse import urlparse
import ipaddress

try:
    from phishing_detector.feature_extractor import (
        MODEL_FEATURES,
        extract_url_features,
        prepare_data,
    )
    from phishing_detector.heuristics import score_url_heuristics
except ImportError:
    from feature_extractor import (
        MODEL_FEATURES,
        extract_url_features,
        prepare_data,
    )
    from heuristics import score_url_heuristics


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")


# ============================================================
# MODEL CACHE
# ============================================================

_model = None
_feature_names = None


def load_model():
    """Loads and caches the trained phishing detection ensemble."""
    global _model, _feature_names

    if _model is not None and _feature_names is not None:
        return _model, _feature_names

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file not found at: {FEATURES_PATH}")

    _model = joblib.load(MODEL_PATH)
    _feature_names = joblib.load(FEATURES_PATH)

    return _model, _feature_names


# Preload model on startup
try:
    load_model()
    print("Phishing detection ensemble loaded successfully!")
except Exception as e:
    print(f"Warning: Could not preload model: {e}")


# ============================================================
# LAYER 1 — KNOWN TRUSTED DOMAINS (ALLOWLIST)
# ============================================================

TRUSTED_DOMAINS = {
    "google.com", "google.co.in", "google.co.uk", "google.ca", "google.com.au",
    "youtube.com", "youtu.be",
    "wikipedia.org", "wikimedia.org",
    "github.com", "gitlab.com", "bitbucket.org",
    "microsoft.com", "live.com", "office.com", "outlook.com", "bing.com",
    "apple.com", "icloud.com",
    "amazon.com", "amazon.in", "amazon.co.uk", "aws.amazon.com",
    "netflix.com", "spotify.com",
    "facebook.com", "instagram.com", "whatsapp.com", "meta.com",
    "twitter.com", "x.com", "linkedin.com", "reddit.com",
    "yahoo.com", "duckduckgo.com",
    "stackoverflow.com", "stackexchange.com",
    "cloudflare.com", "zoom.us", "adobe.com",
    "dropbox.com", "quora.com", "medium.com",
    "nytimes.com", "bbc.com", "cnn.com",
    "paypal.com", "ebay.com", "walmart.com", "target.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com",
    "python.org", "nodejs.org", "reactjs.org", "vuejs.org",
}

# Known malicious or testing domains — always flag as phishing
BLOCKLIST_DOMAINS = {
    "malware.testing.google.test",
    "phishing.test",
}


def _get_root_domain(domain: str) -> str:
    """Strip www. and return root domain."""
    dom = domain.lower()
    if dom.startswith("www."):
        dom = dom[4:]
    return dom


def is_trusted_domain(domain: str) -> bool:
    """Returns True if domain is a verified trusted root or subdomain of one."""
    dom = _get_root_domain(domain)
    if dom in TRUSTED_DOMAINS:
        return True
    for td in TRUSTED_DOMAINS:
        if dom.endswith("." + td):
            return True
    return False


def is_blocklisted_domain(domain: str) -> bool:
    """Returns True if domain is in the known-phishing blocklist."""
    dom = _get_root_domain(domain)
    return dom in BLOCKLIST_DOMAINS


# ============================================================
# LAYER 5 — SCORE FUSION + RISK CALIBRATION
# ============================================================

def _fuse_scores(ml_phish_prob: float, h_score: float) -> float:
    """
    Combine ML ensemble probability and heuristic score into a single
    phishing probability using a weighted average.

    Weights: ML ensemble gets 65%, heuristics 35%.
    This means strong heuristic signals (punycode, IP, etc.) can override
    a borderline ML decision, but the ML model remains dominant.
    """
    fused = 0.65 * ml_phish_prob + 0.35 * h_score
    return round(max(0.0, min(1.0, fused)), 4)


def _map_risk_tier(phish_prob: float, is_trusted: bool, is_ip: bool):
    """
    Map phishing probability to a risk tier and result label.

    Tiers:
      0.00 – 0.30 → LEGITIMATE / LOW
      0.30 – 0.55 → SUSPICIOUS / MEDIUM
      0.55 – 0.80 → PHISHING   / HIGH
      0.80 – 1.00 → PHISHING   / CRITICAL
    """
    if is_ip:
        return "PHISHING", "CRITICAL"

    if is_trusted and phish_prob < 0.35:
        return "LEGITIMATE", "LOW"

    if phish_prob < 0.30:
        return "LEGITIMATE", "LOW"
    elif phish_prob < 0.55:
        return "SUSPICIOUS", "MEDIUM"
    elif phish_prob < 0.80:
        return "PHISHING", "HIGH"
    else:
        return "PHISHING", "CRITICAL"


def _build_message(result: str, risk_level: str, risk_factors: list) -> str:
    """Generate a human-readable summary message."""
    base = {
        ("LEGITIMATE", "LOW"): "This URL appears safe with no significant phishing indicators detected.",
        ("SUSPICIOUS", "MEDIUM"): "This URL has some suspicious characteristics. Proceed with caution.",
        ("PHISHING", "HIGH"): "High risk: Multiple phishing indicators detected. Avoid submitting credentials.",
        ("PHISHING", "CRITICAL"): "Critical threat: This URL exhibits strong phishing patterns. Do not open.",
    }
    msg = base.get((result, risk_level), "Analysis complete.")
    if risk_factors:
        top = risk_factors[0]
        msg += f" Primary concern: {top}"
    return msg


# ============================================================
# MAIN PREDICT FUNCTION
# ============================================================

def predict_url(url: str) -> dict:
    """
    Hybrid phishing detection with 4 layers:
      1. Allowlist / Blocklist fast-pass
      2. ML Ensemble (30-feature soft-voting)
      3. Deep Lexical Heuristics (17 rule checks)
      4. Score Fusion + Risk-Tier Calibration

    Returns:
        dict with keys: status, url, result, prediction, confidence,
                        confidence_percent, risk_level, message,
                        risk_factors, method_scores, features
    """
    url_clean = str(url).strip()

    if not url_clean:
        return {
            "status": "error",
            "url": "",
            "result": "INVALID",
            "prediction": None,
            "confidence": 0.0,
            "confidence_percent": "0.0%",
            "risk_level": "UNKNOWN",
            "message": "Please provide a valid, non-empty URL to analyze.",
            "risk_factors": [],
            "method_scores": {},
            "features": {},
        }

    try:
        # ---- Parse domain ----
        parsed = urlparse(url_clean if "://" in url_clean else "http://" + url_clean)
        domain = parsed.netloc.lower()
        if "@" in domain:
            domain = domain.split("@")[-1]
        if ":" in domain:
            domain = domain.split(":")[0]

        # ---- Layer 1: Fast-pass checks ----
        trusted = is_trusted_domain(domain)
        blocklisted = is_blocklisted_domain(domain)

        # ---- Extract features (30) ----
        features_dict = extract_url_features(url_clean)

        is_ip = features_dict.get("IsDomainIP", 0) == 1
        is_https = features_dict.get("ISHTTPS", 0) == 1

        # Immediate hard blocks
        if blocklisted:
            return {
                "status": "success",
                "url": url_clean,
                "result": "PHISHING",
                "prediction": 1,
                "confidence": 99.0,
                "confidence_percent": "99.0%",
                "risk_level": "CRITICAL",
                "message": "This domain is on a known phishing blocklist.",
                "risk_factors": ["Domain is in the known phishing blocklist"],
                "method_scores": {"layer1_blocklist": 1.0},
                "features": features_dict,
            }

        # ---- Layer 2: ML Ensemble ----
        model, feature_names = load_model()
        df_single = pd.DataFrame([features_dict])
        X = df_single.reindex(columns=feature_names, fill_value=0)

        raw_pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]

        # Map class index → phishing probability
        # In PhiUSIIL: label 0 = phishing, label 1 = legitimate
        class_indices = {c: idx for idx, c in enumerate(model.classes_)}
        prob_legit = float(proba[class_indices.get(1, 1)])
        prob_phish = float(proba[class_indices.get(0, 0)])

        ml_phish_prob = prob_phish

        # ---- Layer 3: Deep Heuristics ----
        h_score, risk_factors = score_url_heuristics(url_clean, features_dict)

        # ---- Layer 1 allowlist override: trusted + HTTPS → clamp ML toward safe ----
        if trusted and is_https:
            # Reduce phishing probability significantly for trusted domains
            ml_phish_prob = min(ml_phish_prob, 0.15)
            h_score = min(h_score, 0.15)
            if not risk_factors:
                risk_factors = []

        # ---- Layer 4: Score Fusion ----
        fused_prob = _fuse_scores(ml_phish_prob, h_score)
        result, risk_level = _map_risk_tier(fused_prob, trusted, is_ip)

        # Convert phishing probability to confidence in that result
        if result == "LEGITIMATE":
            confidence = round((1.0 - fused_prob) * 100, 1)
        else:
            confidence = round(fused_prob * 100, 1)

        # For trusted+HTTPS force a minimum confidence for legitimate results
        if trusted and is_https and result == "LEGITIMATE":
            confidence = max(confidence, 97.0)

        message = _build_message(result, risk_level, risk_factors)

        method_scores = {
            "ml_phish_probability": round(ml_phish_prob, 4),
            "heuristic_score": h_score,
            "fused_phish_probability": fused_prob,
            "is_trusted_domain": trusted,
        }

        return {
            "status": "success",
            "url": url_clean,
            "result": result,
            "prediction": 1 if result == "PHISHING" else (2 if result == "SUSPICIOUS" else 0),
            "confidence": confidence,
            "confidence_percent": f"{confidence:.1f}%",
            "risk_level": risk_level,
            "message": message,
            "risk_factors": risk_factors,
            "method_scores": method_scores,
            "features": features_dict,
        }

    except Exception as exc:
        return {
            "status": "error",
            "url": url_clean,
            "result": "ERROR",
            "prediction": None,
            "confidence": 0.0,
            "confidence_percent": "0.0%",
            "risk_level": "UNKNOWN",
            "message": f"Prediction failed: {str(exc)}",
            "risk_factors": [],
            "method_scores": {},
            "features": {},
        }


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("       HYBRID PHISHING URL DETECTOR")
    print("=" * 55)

    test_input = input("\nEnter a URL to check (or press Enter for default): ").strip()
    test_input = test_input or "https://www.google.com"

    res = predict_url(test_input)

    print("\n" + "=" * 55)
    print("URL         :", res["url"])
    print("RESULT      :", res["result"])
    print("RISK LEVEL  :", res["risk_level"])
    print("CONFIDENCE  :", res["confidence_percent"])
    print("MESSAGE     :", res["message"])

    if res.get("risk_factors"):
        print("\nRisk Factors:")
        for rf in res["risk_factors"]:
            print(f"  ⚠  {rf}")

    if res.get("method_scores"):
        print("\nMethod Scores (transparency):")
        for k, v in res["method_scores"].items():
            print(f"  {k:30}: {v}")

    print("\nTop Features:")
    for k, v in list(res["features"].items())[:10]:
        print(f"  {k:30}: {v}")
    print("=" * 55)