import os
import cv2
from urllib.parse import urlparse


def is_valid_url(text):
    """
    Checks if the decoded text looks like a URL.
    """
    text = str(text).strip()
    if not text:
        return False
    if text.startswith(("http://", "https://", "ftp://", "www.")):
        return True
    try:
        parsed = urlparse(text)
        return bool(parsed.scheme and parsed.netloc) or ("." in text and " " not in text)
    except Exception:
        return False


def decode_qr(image_path):
    """
    Reads a QR code from an image file and returns the decoded payload.
    Uses multi-stage decoding (standard -> grayscale -> adaptive threshold).
    """
    if not os.path.exists(image_path):
        return {
            "success": False,
            "data": "",
            "is_url": False,
            "message": "QR image file not found on server."
        }

    try:
        image = cv2.imread(image_path)

        if image is None:
            return {
                "success": False,
                "data": "",
                "is_url": False,
                "message": "Unable to read the uploaded image file (unsupported or corrupted format)."
            }

        detector = cv2.QRCodeDetector()

        # Stage 1: Standard color image detection
        data, points, _ = detector.detectAndDecode(image)

        # Stage 2: Grayscale fallback
        if not data:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            data, points, _ = detector.detectAndDecode(gray)

        # Stage 3: Adaptive threshold fallback
        if not data:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0
            )
            data, points, _ = detector.detectAndDecode(thresh)

        if not data:
            return {
                "success": False,
                "data": "",
                "is_url": False,
                "message": "No readable QR code found in the uploaded image. Please ensure the QR code is clear and well-lit."
            }

        data_clean = str(data).strip()
        is_url = is_valid_url(data_clean)

        return {
            "success": True,
            "data": data_clean,
            "is_url": is_url,
            "message": "QR code successfully decoded."
        }

    except Exception as err:
        return {
            "success": False,
            "data": "",
            "is_url": False,
            "message": f"Error decoding QR image: {str(err)}"
        }


if __name__ == "__main__":
    print("QR Code Detector is ready.")