from __future__ import annotations

import uuid

from flask import Flask, jsonify, render_template, request

from neurovision import build_default_pipeline

app = Flask(__name__)
pipeline = build_default_pipeline()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/neurovision/run")
def run_neurovision_pipeline():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file. Send multipart/form-data with 'image'."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded image is empty."}), 400

    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    response = pipeline.run(image_bytes=image_bytes, request_id=request_id)
    return jsonify(response.to_dict()), 200
