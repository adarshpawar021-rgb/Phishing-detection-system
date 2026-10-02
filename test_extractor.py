import os
import joblib
import pandas as pd
from urllib.parse import urlparse
import ipaddress

model = joblib.load('models/phishing_model.pkl')
features = joblib.load('models/model_features.pkl')

def extract_phi_url(url):
    url_clean = str(url).strip()
    if not url_clean:
        return {f: 0 for f in features}

    parsed = urlparse(url_clean)
    if not parsed.netloc:
        parsed = urlparse("http://" + url_clean)
        
    domain = parsed.netloc.lower()
    if "@" in domain:
        domain = domain.split("@")[-1]
    if ":" in domain:
        domain = domain.split(":")[0]

    # Clean domain without www. for core character calculations matching PhiUSIIL
    dom_core = domain[4:] if domain.startswith("www.") else domain
    parts = domain.split(".")
    tld = parts[-1] if len(parts) >= 2 else ""
    tld_len = len(tld)
    
    # Core string: domain without www. + path + query
    path_query = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")
    core_str = dom_core + path_query

    url_len = len(url_clean)
    
    # Letters: in core string, TLD contributes (tld_len - 1) letters matching dataset encoding
    # E.g. radware (7) + co (2) = 9
    domain_body = dom_core[: -len(tld) - 1] if len(parts) >= 2 and len(dom_core) > len(tld) + 1 else dom_core
    letters = sum(1 for c in domain_body if c.isalpha()) + max(tld_len - 1, 0) + sum(1 for c in path_query if c.isalpha())
    letter_ratio = (letters / url_len) if url_len > 0 else 0.0

    digits = sum(1 for c in core_str if c.isdigit())
    digit_ratio = (digits / url_len) if url_len > 0 else 0.0

    # Query separators
    eq = url_clean.count("=")
    qm = url_clean.count("?")
    amp = url_clean.count("&")

    # Special characters in core string (excluding =, ?, &)
    spec_chars = set("!#$%&'()*+,-/:;=?@[]^_`{|}~.")
    other_spec = sum(1 for c in core_str if c in spec_chars and c not in ["=", "?", "&"])
    tot_spec = eq + qm + amp + other_spec
    spec_ratio = (tot_spec / url_len) if url_len > 0 else 0.0

    # Obfuscation
    obf_count = url_clean.count("%") + url_clean.count("@")
    has_obf = 1 if obf_count > 0 else 0
    obf_ratio = (obf_count / url_len) if url_len > 0 else 0.0

    # IP check
    try:
        ipaddress.ip_address(domain)
        is_ip = 1
    except ValueError:
        is_ip = 0

    # Subdomains (excluding www)
    no_subdomain = max(len(parts) - 2, 0)

    # Keywords
    url_lower = url_clean.lower()
    bank_keywords = ["bank", "banking", "account", "netbanking", "creditcard", "debitcard"]
    pay_keywords = ["pay", "payment", "checkout", "billing", "invoice", "paypal"]
    crypto_keywords = ["crypto", "bitcoin", "ethereum", "wallet", "usdt", "binance"]

    bank = int(any(k in url_lower for k in bank_keywords))
    pay = int(any(k in url_lower for k in pay_keywords))
    crypto = int(any(k in url_lower for k in crypto_keywords))

    is_https = 1 if parsed.scheme.lower() == "https" else 0

    return {
        "URLLength": url_len,
        "DomainLength": len(domain),
        "IsDomainIP": is_ip,
        "TLDLength": tld_len,
        "NoOfSubDomain": no_subdomain,
        "HasObfuscation": has_obf,
        "NoofobfuscatedChar": obf_count,
        "ObfuscationRatio": round(obf_ratio, 6),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": round(letter_ratio, 6),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": round(digit_ratio, 6),
        "NoOfEqualsInURL": eq,
        "NoOFQMarkInURL": qm,
        "NoOfAmpersandInURL": amp,
        "NoOfOtherSpecialCharsInURL": other_spec,
        "SpacialCharRatioInURL": round(spec_ratio, 6),
        "ISHTTPS": is_https,
        "Bank": bank,
        "Pay": pay,
        "Crypto": crypto
    }


test_urls = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.wikipedia.org",
    "https://github.com",
    "https://www.microsoft.com",
    "https://www.amazon.com",
    "https://www.southbankmosaics.com",
    "http://192.168.1.1/login",
    "http://paypal-security-update-account.xyz/login",
    "http://apple-id-verify-locked-account.net/login.php",
    "http://secure-login-verify-account.com"
]

df = pd.DataFrame([extract_phi_url(u) for u in test_urls])[features]
preds = model.predict(df)
probs = model.predict_proba(df)

print(f"{'URL':52} | {'RESULT':20} | {'LEGIT CONF':10} | {'PHISH CONF':10}")
print("-" * 102)
for u, p, pr in zip(test_urls, preds, probs):
    legit_conf = f"{pr[1]*100:.1f}%"
    phish_conf = f"{pr[0]*100:.1f}%"
    res_str = "LEGITIMATE (Safe)" if p == 1 else "PHISHING (Fake)"
    print(f"{u:52} | {res_str:20} | {legit_conf:10} | {phish_conf:10}")
