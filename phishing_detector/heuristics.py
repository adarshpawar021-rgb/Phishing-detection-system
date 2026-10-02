"""
heuristics.py — Deep Lexical Heuristics Engine

Provides a rule-based risk scoring layer that complements the ML ensemble.
Each rule returns a risk delta (0.0 – 1.0) and a human-readable reason string.
The final output is a combined heuristic score and a list of triggered risk factors.

Usage:
    from phishing_detector.heuristics import score_url_heuristics
    h_score, risk_factors = score_url_heuristics(url, features_dict)
"""

import re
import math
from urllib.parse import urlparse


# ============================================================
# CONSTANTS
# ============================================================

# TLDs historically associated with free/abusive registrations
SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "top", "click", "loan", "gq",
    "pw", "cc", "su", "ws", "biz", "info", "mobi", "name", "online",
    "site", "space", "tech", "live", "store", "shop", "icu", "buzz",
    "work", "fun", "uno", "cyou", "rest", "world", "monster"
}

# High-value impersonation targets; hyphens around these in the domain are suspicious
BRAND_TARGETS = {
    "paypal", "apple", "amazon", "google", "facebook", "instagram",
    "netflix", "microsoft", "yahoo", "ebay", "twitter", "linkedin",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hdfc", "sbi",
    "icici", "axisbank", "steam", "roblox", "coinbase", "binance",
    "dropbox", "adobe", "office365", "onedrive",
}

# Legitimate TLDs that rarely host phishing
TRUSTED_TLDS = {"gov", "edu", "mil"}

# Phishing-specific path keywords
PHISH_PATH_KEYWORDS = [
    "login", "signin", "secure", "verify", "validation", "update",
    "confirm", "account", "webscr", "cmd=_s-xclick", "password",
    "credential", "auth", "authenticate", "support", "service",
    "recover", "unlock", "suspend", "urgent", "alert",
]

# ============================================================
# HELPER: SHANNON ENTROPY
# ============================================================

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ============================================================
# INDIVIDUAL HEURISTIC CHECKS
# (each returns: triggered: bool, weight: float, reason: str)
# ============================================================

def _check_ip_address(features: dict):
    if features.get("IsDomainIP", 0) == 1:
        return True, 0.95, "Domain is a raw IP address — never used by legitimate sites"
    return False, 0.0, ""


def _check_no_https(features: dict):
    if features.get("ISHTTPS", 0) == 0:
        return True, 0.30, "No HTTPS — connection is unencrypted"
    return False, 0.0, ""


def _check_suspicious_tld(features: dict, tld: str):
    if tld.lower() in SUSPICIOUS_TLDS:
        return True, 0.40, f"Suspicious free TLD (.{tld}) commonly abused in phishing"
    return False, 0.0, ""


def _check_trusted_tld(tld: str):
    """Trusted TLDs (gov, edu, mil) reduce risk."""
    if tld.lower() in TRUSTED_TLDS:
        return True, -0.50, f"Trusted government/education TLD (.{tld})"
    return False, 0.0, ""


def _check_punycode(features: dict):
    if features.get("IsPunycode", 0) == 1:
        return True, 0.70, "Punycode/homograph domain detected — likely visual spoofing attack"
    return False, 0.0, ""


def _check_url_entropy(features: dict):
    entropy = features.get("URLEntropy", 0.0)
    if entropy > 4.5:
        return True, 0.35, f"Very high URL entropy ({entropy:.2f}) — suggests random/obfuscated string"
    return False, 0.0, ""


def _check_domain_entropy(features: dict, domain: str):
    d_entropy = features.get("DomainEntropy", 0.0)
    if d_entropy > 3.8 and len(domain) > 12:
        return True, 0.30, f"High domain entropy ({d_entropy:.2f}) — domain may be auto-generated (DGA)"
    return False, 0.0, ""


def _check_long_url(features: dict):
    url_len = features.get("URLLength", 0)
    if url_len > 100:
        return True, 0.15, f"Unusually long URL ({url_len} chars) — may contain embedded redirect"
    return False, 0.0, ""


def _check_excessive_subdomains(features: dict):
    n_sub = features.get("NoOfSubDomain", 0)
    if n_sub >= 4:
        return True, 0.40, f"Excessive subdomain depth ({n_sub}) — typical phishing obfuscation"
    elif n_sub == 3:
        return True, 0.15, f"High subdomain count ({n_sub}) — slightly suspicious"
    return False, 0.0, ""


def _check_brand_in_path(features: dict, path: str, domain: str):
    """Brand name present in path/query but not in domain → impersonation."""
    if features.get("HasBrandMismatch", 0) == 1:
        return True, 0.60, "Brand keyword in URL path but not in domain — likely impersonation"
    return False, 0.0, ""


def _check_brand_in_subdomain(domain: str):
    """Brand name present in subdomain (e.g. paypal.evil.com)."""
    subdomains = domain.split(".")[:-2]  # everything before SLD.TLD
    for brand in BRAND_TARGETS:
        for sub in subdomains:
            if brand in sub.lower():
                return True, 0.75, f"Brand name '{brand}' found in subdomain — typical phishing pattern"
    return False, 0.0, ""


def _check_domain_hyphens(features: dict, domain: str):
    hyphens = features.get("DomainHyphenCount", 0)
    if hyphens >= 3:
        return True, 0.30, f"Many hyphens ({hyphens}) in domain — e.g. paypal-secure-login style"
    return False, 0.0, ""


def _check_phish_path_keywords(url_lower: str):
    triggered_kw = [kw for kw in PHISH_PATH_KEYWORDS if kw in url_lower]
    if len(triggered_kw) >= 3:
        return True, 0.50, f"Multiple phishing keywords in URL: {', '.join(triggered_kw[:5])}"
    elif len(triggered_kw) >= 1:
        return True, 0.20, f"Phishing keyword in URL: {', '.join(triggered_kw[:3])}"
    return False, 0.0, ""


def _check_redirect_in_url(features: dict):
    if features.get("HasRedirectInURL", 0) == 1:
        return True, 0.65, "URL embeds another URL — classic open-redirect phishing technique"
    return False, 0.0, ""


def _check_obfuscation(features: dict):
    if features.get("HasObfuscation", 0) == 1:
        ratio = features.get("ObfuscationRatio", 0.0)
        return True, min(0.20 + ratio * 2, 0.60), "URL contains percent-encoding or @ obfuscation"
    return False, 0.0, ""


def _check_suspicious_keywords(features: dict, url_lower: str):
    """Financial keyword present without HTTPS is highly suspicious."""
    has_fin = (
        features.get("Bank", 0)
        + features.get("Pay", 0)
        + features.get("Crypto", 0)
    ) > 0
    is_https = features.get("ISHTTPS", 0) == 1
    if has_fin and not is_https:
        return True, 0.55, "Financial keyword (bank/pay/crypto) present but no HTTPS"
    return False, 0.0, ""


def _check_digit_heavy_domain(domain: str):
    """Domains with many digits are suspicious (e.g. secure12345bank.com)."""
    sld = domain.split(".")[-2] if len(domain.split(".")) >= 2 else domain
    digit_count = sum(1 for c in sld if c.isdigit())
    if digit_count >= 4:
        return True, 0.25, f"Domain SLD contains many digits ({digit_count}) — suspicious pattern"
    return False, 0.0, ""


# ============================================================
# MAIN HEURISTIC SCORER
# ============================================================

def score_url_heuristics(url: str, features: dict):
    """
    Run all heuristic checks and return:
    - heuristic_score (float 0.0–1.0): probability of phishing from rules alone
    - risk_factors (list[str]): human-readable list of triggered reasons

    The score is a weighted average of triggered checks, clamped to [0, 1].
    Negative weights (e.g. trusted TLD) can reduce the final score.
    """
    url_clean = str(url).strip()
    if not url_clean:
        return 0.5, []

    # Parse for check helpers
    parsed = urlparse(url_clean if "://" in url_clean else "http://" + url_clean)
    domain = parsed.netloc.lower()
    if "@" in domain:
        domain = domain.split("@")[-1]
    if ":" in domain:
        domain = domain.split(":")[0]

    dom_core = domain[4:] if domain.startswith("www.") else domain
    parts = domain.split(".") if domain else []
    tld = parts[-1] if len(parts) >= 2 else ""
    path = (parsed.path or "").lower()
    url_lower = url_clean.lower()

    # Collect all checks
    all_checks = [
        _check_ip_address(features),
        _check_no_https(features),
        _check_suspicious_tld(features, tld),
        _check_trusted_tld(tld),
        _check_punycode(features),
        _check_url_entropy(features),
        _check_domain_entropy(features, dom_core),
        _check_long_url(features),
        _check_excessive_subdomains(features),
        _check_brand_in_path(features, path, dom_core),
        _check_brand_in_subdomain(domain),
        _check_domain_hyphens(features, dom_core),
        _check_phish_path_keywords(url_lower),
        _check_redirect_in_url(features),
        _check_obfuscation(features),
        _check_suspicious_keywords(features, url_lower),
        _check_digit_heavy_domain(dom_core),
    ]

    risk_factors = []
    total_weight = 0.0
    positive_weight = 0.0  # sum of all possible positive check weights for normalization

    for triggered, weight, reason in all_checks:
        positive_weight += abs(weight)
        if triggered:
            total_weight += weight
            if reason:
                risk_factors.append(reason)

    # Normalize to [0, 1]; negative weights (trusted TLD) reduce the score
    if positive_weight > 0:
        raw_score = total_weight / positive_weight
    else:
        raw_score = 0.0

    # Shift to [0, 1] range (raw_score can be negative for very safe URLs)
    heuristic_score = max(0.0, min(1.0, (raw_score + 0.5) * 0.9))

    return round(heuristic_score, 4), risk_factors


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":
    try:
        from phishing_detector.feature_extractor import extract_url_features
    except ImportError:
        from feature_extractor import extract_url_features

    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login",
        "http://paypal-secure-verify.xyz/account/confirm",
        "https://www.paypal.com/signin",
        "http://bank.secure-login.update-account.com/auth",
    ]

    for u in test_urls:
        feats = extract_url_features(u)
        score, factors = score_url_heuristics(u, feats)
        print(f"\nURL: {u}")
        print(f"  Heuristic Score: {score:.4f}")
        print(f"  Risk Factors   : {factors}")
