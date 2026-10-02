from urllib.parse import urlparse
try:
    from phishing_detector.predictor import predict_url
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from phishing_detector.predictor import predict_url


def detect_phishing(url):
    """
    ML-powered URL phishing detector wrapper.
    Delegates to the pre-trained Random Forest model in phishing_detector/predictor.py.
    """
    return predict_url(url)


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "https://www.wikipedia.org",
        "http://192.168.1.10/login",
        "http://secure-login-verify-account.com"
    ]

    for url in test_urls:
        print("\nURL:", url)
        res = detect_phishing(url)
        print("Result:", res["result"])
        print("Prediction:", res["prediction"])
        print("Confidence:", res["confidence_percent"])
        print("Message:", res["message"])