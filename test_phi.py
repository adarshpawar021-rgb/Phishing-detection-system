import os
import joblib
import pandas as pd
from urllib.parse import urlparse

model = joblib.load('models/phishing_model.pkl')
features = joblib.load('models/model_features.pkl')

def extract_phi(url):
    url_clean = str(url).strip()
    parsed = urlparse(url_clean)
    if not parsed.netloc:
        parsed = urlparse("http://" + url_clean)
    
    domain = parsed.netloc
    if "@" in domain:
        domain = domain.split("@")[-1]
    if ":" in domain:
        domain = domain.split(":")[0]
    
    dom_core = domain
    if dom_core.startswith("www."):
        dom_core = dom_core[4:]
        
    parts = domain.split(".")
    tld = parts[-1] if len(parts) >= 2 else ""
    path_query = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")
    core_str = dom_core + path_query
    
    url_len = len(url_clean)
    letters = sum(1 for c in core_str if c.isalpha())
    digits = sum(1 for c in core_str if c.isdigit())
    
    eq = url_clean.count("=")
    qm = url_clean.count("?")
    amp = url_clean.count("&")
    
    spec_chars = set("!#$%&'()*+,-/:;=?@[]^_`{|}~.")
    other_spec = sum(1 for c in core_str if c in spec_chars and c not in ["=", "?", "&"])
    tot_spec = eq + qm + amp + other_spec
    
    is_ip = 0
    try:
        import ipaddress
        ipaddress.ip_address(domain)
        is_ip = 1
    except ValueError:
        is_ip = 0

    return {
        "URLLength": url_len,
        "DomainLength": len(domain),
        "IsDomainIP": is_ip,
        "TLDLength": len(tld),
        "NoOfSubDomain": max(len(parts) - 2, 0),
        "HasObfuscation": 1 if ("%" in url_clean or "@" in url_clean) else 0,
        "NoofobfuscatedChar": url_clean.count("%") + url_clean.count("@"),
        "ObfuscationRatio": (url_clean.count("%") + url_clean.count("@")) / url_len if url_len else 0,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": round(letters / url_len, 6) if url_len else 0,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": round(digits / url_len, 6) if url_len else 0,
        "NoOfEqualsInURL": eq,
        "NoOFQMarkInURL": qm,
        "NoOfAmpersandInURL": amp,
        "NoOfOtherSpecialCharsInURL": other_spec,
        "SpacialCharRatioInURL": round(tot_spec / url_len, 6) if url_len else 0,
        "ISHTTPS": 1 if parsed.scheme.lower() == "https" else 0,
        "Bank": 1 if any(w in url_clean.lower() for w in ["bank", "banking", "account", "netbanking"]) else 0,
        "Pay": 1 if any(w in url_clean.lower() for w in ["pay", "payment", "paypal", "checkout"]) else 0,
        "Crypto": 1 if any(w in url_clean.lower() for w in ["crypto", "bitcoin", "ethereum", "wallet", "binance"]) else 0
    }

test_urls = [
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://www.youtube.com",
    "https://github.com",
    "https://www.amazon.com",
    "http://192.168.1.1/login",
    "http://paypal-security-update-account.xyz/login",
    "http://apple-id-verify-locked-account.net/login.php"
]

df = pd.DataFrame([extract_phi(u) for u in test_urls])[features]
preds = model.predict(df)
probs = model.predict_proba(df)

for u, p, pr in zip(test_urls, preds, probs):
    legit_prob = pr[1] * 100
    phish_prob = pr[0] * 100
    status = "LEGITIMATE (Safe)" if p == 1 else "PHISHING (Fake)"
    print(f"{u:55} => {status:20} [Legit: {legit_prob:5.1f}%, Phish: {phish_prob:5.1f}%]")
