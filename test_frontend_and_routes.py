import os
import io
import unittest
import qrcode
import numpy as np
import cv2
from app import app
from phishing_detector.predictor import predict_url

class TestFrontendAndRoutes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        os.makedirs("test_assets", exist_ok=True)

        # 1. Generate legitimate QR image
        legit_qr = qrcode.make("https://www.google.com")
        cls.legit_qr_path = os.path.join("test_assets", "legit_qr.png")
        legit_qr.save(cls.legit_qr_path)

        # 2. Generate phishing QR image
        phish_qr = qrcode.make("http://paypal-security-update-account.xyz/login")
        cls.phish_qr_path = os.path.join("test_assets", "phish_qr.png")
        phish_qr.save(cls.phish_qr_path)

        # 3. Generate invalid (non-QR) image
        cls.invalid_qr_path = os.path.join("test_assets", "invalid_image.png")
        blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(cls.invalid_qr_path, blank_img)

    def test_01_home_page_rendering(self):
        """Test GET / loads home dashboard correctly with all required sections."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("PhishGuard AI", html)
        self.assertIn("Phishing URL Scanner", html)
        self.assertIn("Fake QR Code Scanner", html)
        self.assertIn("4-Layer Hybrid Detection Architecture", html)
        self.assertIn("Comprehensive 30-Feature Vector Space", html)
        self.assertIn("Final Year Engineering Project", html)

    def test_02_legitimate_url_scan(self):
        """Test POST / with a legitimate URL."""
        resp = self.client.post("/", data={"url": "https://www.google.com"})
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("LEGITIMATE WEBSITE", html)
        self.assertIn("LOW RISK", html)
        self.assertIn("google.com", html)
        self.assertIn("ML Soft-Voting Score", html)
        self.assertIn("Inspect 30 Extracted ML Feature Vectors", html)

    def test_03_suspicious_phishing_url_scan(self):
        """Test POST / with a suspicious/phishing URL."""
        resp = self.client.post("/", data={"url": "http://paypal-security-update-account.xyz/login"})
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertTrue("PHISHING" in html or "SUSPICIOUS" in html)
        self.assertIn("paypal-security-update-account.xyz", html)
        self.assertIn("Identified Phishing Threat Signals", html)

    def test_04_invalid_empty_url_scan(self):
        """Test POST / with invalid or empty URL."""
        resp = self.client.post("/", data={"url": ""})
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("PhishGuard AI", html)

    def test_05_legitimate_qr_scan(self):
        """Test POST /scan-qr with a legitimate QR image."""
        with open(self.legit_qr_path, "rb") as f:
            data = {"qr_image": (f, "legit_qr.png")}
            resp = self.client.post("/scan-qr", data=data, content_type="multipart/form-data")
        
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("LEGITIMATE QR CODE", html)
        self.assertIn("https://www.google.com", html)
        self.assertIn("Decoded QR Payload Data", html)

    def test_06_suspicious_phishing_qr_scan(self):
        """Test POST /scan-qr with a suspicious/phishing QR image."""
        with open(self.phish_qr_path, "rb") as f:
            data = {"qr_image": (f, "phish_qr.png")}
            resp = self.client.post("/scan-qr", data=data, content_type="multipart/form-data")
        
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertTrue("QUISHING" in html or "PHISHING" in html or "SUSPICIOUS" in html)
        self.assertIn("paypal-security-update-account.xyz", html)
        self.assertIn("Decoded QR Payload Data", html)

    def test_07_invalid_non_qr_image_scan(self):
        """Test POST /scan-qr with a non-QR image."""
        with open(self.invalid_qr_path, "rb") as f:
            data = {"qr_image": (f, "invalid_image.png")}
            resp = self.client.post("/scan-qr", data=data, content_type="multipart/form-data")
        
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("QR CODE DECODING FAILED", html)
        self.assertIn("No readable QR code found", html)

    @classmethod
    def tearDownClass(cls):
        # Clean up test files
        for p in [cls.legit_qr_path, cls.phish_qr_path, cls.invalid_qr_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if os.path.exists("test_assets"):
            try:
                os.rmdir("test_assets")
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
