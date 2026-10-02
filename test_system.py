import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from phishing_detector.feature_extractor import MODEL_FEATURES, extract_url_features, prepare_data
from phishing_detector.predictor import predict_url, load_model
from qr_detector.qr_detector import decode_qr, is_valid_url
from app import app


class TestPhishingDetectionSystem(unittest.TestCase):

    def test_01_feature_list_and_count(self):
        """Verify the 30 features match models/model_features.pkl exactly."""
        model, feature_names = load_model()
        self.assertEqual(len(feature_names), 30, "Model feature count must be exactly 30")
        self.assertEqual(list(feature_names), MODEL_FEATURES, "Feature names and order must match MODEL_FEATURES")

    def test_02_feature_extractor_values(self):
        """Verify feature extraction returns all 30 features with expected types."""
        url = "https://www.google.com/search?q=test"
        features = extract_url_features(url)
        self.assertEqual(len(features), 30)
        for feat in MODEL_FEATURES:
            self.assertIn(feat, features, f"Missing feature {feat}")
            self.assertIsInstance(features[feat], (int, float), f"Feature {feat} must be numeric")

        self.assertEqual(features["ISHTTPS"], 1)
        self.assertEqual(features["NoOFQMarkInURL"], 1)

    def test_03_prepare_data_no_label_error(self):
        """Verify prepare_data on single URL dataframe does not raise label not found error."""
        df = pd.DataFrame({"URL": ["https://example.com"]})
        X, y = prepare_data(df)
        self.assertEqual(X.shape, (1, 30))
        self.assertIsNone(y)

    def test_04_predictor_legitimate_url(self):
        """Verify predict_url on known legitimate URLs returns LEGITIMATE with risk_factors key."""
        res = predict_url("https://www.google.com")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], "LEGITIMATE")
        self.assertEqual(res["prediction"], 0)
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertEqual(len(res["features"]), 30)
        self.assertIn("risk_factors", res)
        self.assertIn("method_scores", res)

    def test_04b_predictor_phishing_url(self):
        """Verify predict_url on a known suspicious phishing pattern URL."""
        res = predict_url("http://paypal-security-update-account.xyz/login")
        self.assertEqual(res["status"], "success")
        self.assertIn(res["result"], ["PHISHING", "SUSPICIOUS"])
        self.assertGreaterEqual(res["confidence"], 50.0)

    def test_05_predictor_empty_input(self):
        """Verify predict_url handles empty or whitespace input gracefully."""
        res = predict_url("   ")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["result"], "INVALID")
        self.assertIsNone(res["prediction"])

    def test_06_flask_get_home(self):
        """Verify GET / returns 200 OK and renders the dashboard."""
        client = app.test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PhishGuard", response.data)
        self.assertIn(b"URL Scanner", response.data)
        self.assertIn(b"QR Code Scanner", response.data)

    def test_07_flask_post_url_scan(self):
        """Verify POST / scans a URL and renders the result."""
        client = app.test_client()
        response = client.post("/", data={"url": "https://www.google.com"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"google.com", response.data)
        self.assertIn(b"Confidence", response.data)

    def test_08_flask_api_scan_url(self):
        """Verify JSON API endpoint POST /api/scan-url."""
        client = app.test_client()
        response = client.post("/api/scan-url", json={"url": "https://www.wikipedia.org"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("result", data)

    def test_09_qr_url_validator(self):
        """Verify QR URL format checker."""
        self.assertTrue(is_valid_url("https://www.google.com"))
        self.assertTrue(is_valid_url("http://example.com/test"))
        self.assertTrue(is_valid_url("www.testsite.org"))
        self.assertFalse(is_valid_url(""))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RUNNING UNIT AND INTEGRATION TESTS FOR PHISHING SYSTEM")
    print("=" * 60 + "\n")
    unittest.main(verbosity=2)
