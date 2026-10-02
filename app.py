import os
import uuid
from flask import Flask, render_template, request, jsonify
from phishing_detector.predictor import predict_url
from qr_detector.qr_detector import decode_qr

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():
    url_result = None
    qr_result = None
    active_tab = "url"

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url:
            url_result = predict_url(url)
            active_tab = "url"

    return render_template(
        "index.html",
        result=url_result,
        qr_result=qr_result,
        active_tab=active_tab
    )


@app.route("/scan-qr", methods=["POST"])
def scan_qr():
    url_result = None
    qr_result = None
    active_tab = "qr"

    if "qr_image" not in request.files:
        return render_template(
            "index.html",
            result=None,
            qr_result={
                "success": False,
                "status": "error",
                "message": "No QR image file was provided in upload."
            },
            active_tab="qr"
        )

    file = request.files["qr_image"]

    if file.filename == "":
        return render_template(
            "index.html",
            result=None,
            qr_result={
                "success": False,
                "status": "error",
                "message": "Please select a QR code image to upload."
            },
            active_tab="qr"
        )

    # Save temporary file
    ext = os.path.splitext(file.filename)[1] or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)

        # Decode QR code
        decoded = decode_qr(filepath)

        if not decoded["success"]:
            qr_result = {
                "success": False,
                "status": "error",
                "message": decoded["message"]
            }
        else:
            qr_data = decoded["data"]
            # Run ML phishing detector on the extracted QR payload
            phishing_analysis = predict_url(qr_data)

            qr_result = {
                "success": True,
                "status": "success",
                "data": qr_data,
                "is_url": decoded.get("is_url", True),
                "phishing": phishing_analysis
            }

    finally:
        # Clean up temporary uploaded file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

    return render_template(
        "index.html",
        result=url_result,
        qr_result=qr_result,
        active_tab="qr"
    )


# ============================================================
# OPTIONAL JSON API ENDPOINTS (FOR ASYNC OR API CLIENTS)
# ============================================================

@app.route("/api/scan-url", methods=["POST"])
def api_scan_url():
    data = request.get_json(silent=True) or request.form
    url = data.get("url", "").strip() if data else ""

    if not url:
        return jsonify({
            "status": "error",
            "message": "Missing 'url' parameter"
        }), 400

    result = predict_url(url)
    return jsonify(result)


@app.route("/api/scan-qr", methods=["POST"])
def api_scan_qr():
    if "qr_image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No QR image uploaded"
        }), 400

    file = request.files["qr_image"]
    if not file or file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Empty file uploaded"
        }), 400

    ext = os.path.splitext(file.filename)[1] or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)
        decoded = decode_qr(filepath)

        if not decoded["success"]:
            return jsonify(decoded), 400

        qr_data = decoded["data"]
        phishing_analysis = predict_url(qr_data)

        return jsonify({
            "success": True,
            "data": qr_data,
            "phishing": phishing_analysis
        })
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  PHISHGUARD - Phishing URL & QR Detection System")
    print("  Server starting on http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
