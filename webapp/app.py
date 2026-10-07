import os
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_wtf.csrf import CSRFProtect
from langchain_google_genai import ChatGoogleGenerativeAI
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
configured_model_path = os.getenv("MODEL_PATH", "models/full_model.keras")
MODEL_PATH = Path(configured_model_path)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
CLASS_NAMES = (
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
)
IMAGE_SIZE = (208, 176)  # height, width used by the training notebook
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,50}$")
SYSTEM_PROMPT = (
    "You provide general educational information about Alzheimer's disease. "
    "Do not diagnose users or present the model output as medical advice. "
    "Encourage users to consult a qualified healthcare professional."
)

app = Flask(
    __name__,
    instance_relative_config=True,
    instance_path=str(BASE_DIR / "instance"),
)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true",
)
csrf = CSRFProtect(app)

INSTANCE_DIR = Path(app.instance_path)
UPLOAD_DIR = INSTANCE_DIR / "uploads"
DATABASE_PATH = INSTANCE_DIR / "site.db"
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_model_lock = threading.Lock()


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                date_uploaded TEXT NOT NULL,
                saliency_map TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users (username)
            );
            """
        )


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH.is_file():
                    raise FileNotFoundError(
                        f"Model not found at {MODEL_PATH}. Set MODEL_PATH to a trusted model file."
                    )
                _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(file_path):
    image_bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("The uploaded file is not a readable image.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = IMAGE_SIZE
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def generate_saliency_map(model, image):
    input_image = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        predictions = model(input_image, training=False)
        predicted_label = tf.argmax(predictions[0])
        predicted_score = predictions[0, predicted_label]

    gradients = tape.gradient(predicted_score, input_image)
    if gradients is None:
        raise ValueError("The model did not produce input gradients.")

    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0]
    minimum = tf.reduce_min(saliency)
    maximum = tf.reduce_max(saliency)
    saliency = tf.math.divide_no_nan(saliency - minimum, maximum - minimum)
    return np.uint8(np.clip(saliency.numpy(), 0.0, 1.0) * 255)


def save_saliency_map(saliency, destination):
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_HOT)
    if not cv2.imwrite(str(destination), heatmap):
        raise OSError("Could not save the saliency map.")


def current_user_owns(filename):
    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT 1 FROM uploads
            WHERE username = ? AND (filename = ? OR saliency_map = ?)
            """,
            (session.get("username"), filename, filename),
        ).fetchone()
    return row is not None


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(413)
def upload_too_large(_error):
    flash("Image is too large. The maximum upload size is 10 MB.", "error")
    return redirect(url_for("upload_image")), 303


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db_connection() as connection:
            user = connection.execute(
                "SELECT username, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["username"] = user["username"]
            return redirect(url_for("upload_image"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not USERNAME_PATTERN.fullmatch(username):
            flash("Username must be 3-50 characters using letters, numbers, ., _, or -.", "error")
            return render_template("signup.html"), 400
        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
            return render_template("signup.html"), 400

        try:
            with get_db_connection() as connection:
                connection.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
        except sqlite3.IntegrityError:
            flash("Username already exists.", "error")
            return render_template("signup.html"), 409

        flash("Account created successfully.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if not uploaded_file or not uploaded_file.filename:
            flash("Choose an image to upload.", "error")
            return render_template("upload.html"), 400
        if not allowed_file(uploaded_file.filename):
            flash("Only JPG, JPEG, and PNG images are accepted.", "error")
            return render_template("upload.html"), 400

        safe_original = secure_filename(uploaded_file.filename)
        extension = safe_original.rsplit(".", 1)[1].lower()
        filename = f"{uuid4().hex}.{extension}"
        saliency_filename = f"saliency_{uuid4().hex}.png"
        file_path = UPLOAD_DIR / filename
        saliency_path = UPLOAD_DIR / saliency_filename
        uploaded_file.save(file_path)

        try:
            image = prepare_image(file_path)
            model = get_model()
            predictions = model.predict(np.expand_dims(image, axis=0), verbose=0)
            predicted_class = CLASS_NAMES[int(np.argmax(predictions[0]))]
            saliency = generate_saliency_map(model, image)
            save_saliency_map(saliency, saliency_path)

            uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with get_db_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO uploads
                        (username, filename, prediction, date_uploaded, saliency_map)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session["username"],
                        filename,
                        predicted_class,
                        uploaded_at,
                        saliency_filename,
                    ),
                )
        except (ValueError, OSError, FileNotFoundError) as error:
            file_path.unlink(missing_ok=True)
            saliency_path.unlink(missing_ok=True)
            app.logger.warning("Upload processing failed: %s", error)
            flash("The image could not be processed. Check the server model configuration.", "error")
            return render_template("upload.html"), 400

        return render_template(
            "result.html",
            predicted_class=predicted_class,
            filename=filename,
            saliency_filename=saliency_filename,
        )

    return render_template("upload.html")


@app.route("/files/<path:filename>")
def display_image(filename):
    if "username" not in session:
        return redirect(url_for("login"))
    if Path(filename).name != filename or not current_user_owns(filename):
        abort(404)
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    with get_db_connection() as connection:
        uploads = connection.execute(
            """
            SELECT filename, prediction, date_uploaded, saliency_map
            FROM uploads WHERE username = ? ORDER BY id DESC
            """,
            (session["username"],),
        ).fetchall()

    return render_template("dashboard.html", username=session["username"], uploads=uploads)


@app.route("/chat", methods=["POST"])
def chat():
    if "username" not in session:
        return jsonify({"error": "Authentication required."}), 401

    data = request.get_json(silent=True) or {}
    user_input = str(data.get("message", "")).strip()
    if not user_input or len(user_input) > 2000:
        return jsonify({"error": "Message must contain 1-2000 characters."}), 400

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return jsonify({"error": "Chat is not configured on this server."}), 503

    try:
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            temperature=0.5,
            google_api_key=api_key,
        )
        reply = llm.invoke(
            [("system", SYSTEM_PROMPT), ("human", user_input)]
        )
        return jsonify({"response": str(reply.content)})
    except Exception:
        app.logger.exception("Chat request failed")
        return jsonify({"error": "The chat service is temporarily unavailable."}), 502


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
