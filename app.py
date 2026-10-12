"""Flask entry point for InvoiceAI: upload, ask, extract, and reset."""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

from rag.extraction import extract_invoice_fields
from rag.ingestion import add_documents, build_vector_db, chunk_stats, load_pdf, split_documents
from rag.retrieval import answer_query

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))

app = Flask(__name__)
# Set FLASK_SECRET_KEY in .env for stable signed sessions across restarts.
# A random fallback is safer than a hard-coded secret for local development,but invalidates sessions after restart; set the environment value in deployment.
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("COOKIE_SECURE", "false").lower() == "true"
limiter = Limiter(get_remote_address, app=app, default_limits=["50 per hour"])

# Note: this in-memory state is for a single-process prototype. A multi-worker deployment should move session/document state to durable, tenant-scoped storage.
sessions: dict[str, dict] = {}


def _key() -> str:
    if "uid" not in session:
        session["uid"] = uuid.uuid4().hex
    return session["uid"]


def _get_session() -> dict:
    return sessions.setdefault(
        _key(), {"db": None, "files": [], "all_text": "", "upload_paths": []}
    )


def _valid_pdf(file_storage) -> bool:
    filename = (file_storage.filename or "").strip()
    if not filename or Path(filename).suffix.lower() != ".pdf":
        return False
    # Check the file signature instead of trusting only the client-provided name.
    header = file_storage.stream.read(1024)
    file_storage.stream.seek(0)
    return b"%PDF-" in header


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": f"File is too large. Maximum upload size is {MAX_UPLOAD_MB} MB."}), 413


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "InvoiceAI"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
@limiter.limit("10 per hour")
def upload():
    file = request.files.get("pdf")
    if not file or not _valid_pdf(file):
        return jsonify({"error": "Please upload a valid PDF file."}), 400

    original_name = Path(file.filename).name
    safe_name = secure_filename(original_name) or "invoice.pdf"
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    path = UPLOAD_FOLDER / stored_name
    state = _get_session()

    try:
        file.save(path)
        docs = load_pdf(str(path), source_name=original_name)
        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError("No extractable text was found. This PDF may be scanned; OCR is not enabled in this version.")
        chunks = split_documents(docs)
        if not chunks:
            raise ValueError("No text chunks could be created from this PDF.")
        new_db = add_documents(state["db"], chunks) if state["db"] is not None else build_vector_db(chunks)

        # Update state only after extraction/indexing succeeds.
        state["db"] = new_db
        state["files"].append(original_name)
        state["all_text"] += "\n".join(doc.page_content for doc in docs) + "\n"
        state["upload_paths"].append(str(path))
    except Exception:
        app.logger.exception("Invoice upload/indexing failed")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            app.logger.warning("Could not remove failed upload: %s", path)
        return jsonify({"error": "The PDF could not be processed. Check that it is a readable, text-based PDF and try again."}), 500

    stats = chunk_stats(chunks)
    return jsonify({
        "success": True,
        "doc_name": original_name,
        "pages": len(docs),
        "chunks": stats["count"],
        "avg_chunk_size": stats["avg_size"],
        "total_files": len(state["files"]),
        "all_files": state["files"],
    })


@app.route("/ask", methods=["POST"])
@limiter.limit("15 per hour")
def ask():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    query = data.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Query cannot be empty."}), 400
    state = _get_session()
    if state["db"] is None:
        return jsonify({"error": "No document indexed. Upload a PDF first."}), 400
    try:
        result = answer_query(state["db"], query.strip())
        return jsonify(result)
    except Exception:
        app.logger.exception("Invoice question-answering failed")
        return jsonify({"error": "Unable to answer this question right now. Check the API key/network and try again."}), 500


@app.route("/extract", methods=["POST"])
@limiter.limit("10 per hour")
def extract():
    """Extract invoice fields as JSON from the current session's indexed text."""
    state = _get_session()
    if not state["all_text"].strip():
        return jsonify({"error": "No document indexed. Upload a PDF first."}), 400
    try:
        fields = extract_invoice_fields(state["all_text"])
        if "error" in fields:
            return jsonify(fields), 502
        return jsonify(fields)
    except Exception:
        app.logger.exception("Invoice field extraction failed")
        return jsonify({"error": "Structured extraction failed. Check the API key/network and try again."}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Clear this browser session's in-memory index and stored upload files."""
    key = _key()
    state = sessions.pop(key, None)
    if state:
        for path_string in state.get("upload_paths", []):
            try:
                Path(path_string).unlink(missing_ok=True)
            except OSError:
                app.logger.warning("Could not delete uploaded file: %s", path_string)
    session.pop("uid", None)
    return jsonify({"success": True})


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, use_reloader=False)
