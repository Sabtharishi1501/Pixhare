from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from models import db, Photographer, Guest, Event
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
from threading import Thread
from email.message import EmailMessage
from uuid import uuid4
from datetime import datetime, timedelta
import random, smtplib, qrcode, os, shutil, cv2
from chatbot import get_answer, initialize as init_chatbot


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:5000")


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder='static')
app.config.from_object(Config)

# PostgreSQL for Render production
database_url = os.getenv("DATABASE_URL")

if database_url:
    database_url = database_url.replace("postgres://", "postgresql://")
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB upload limit

db.init_app(app)
app.secret_key = app.config['SECRET_KEY']

# Automatically create database tables
with app.app_context():
    db.create_all()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def allowed_file(filename):
    """Return True only for allowed image extensions."""
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def generate_qr_code(data, filename):
    """Generate and save a QR code image."""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save(filename)


def generate_otp():
    return str(random.randint(100000, 999999))


def preprocess_image(image_path, size=(224, 224)):
    """
    Resize and convert image to RGB so DeepFace receives
    a consistently formatted input, improving match accuracy.
    Returns the processed image path (overwrites a temp copy).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path          # fallback – let DeepFace handle it
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp_path = image_path.replace('.jpg', '_proc.jpg').replace('.png', '_proc.png')
        cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return temp_path
    except Exception as e:
        print(f"[preprocess] Warning: {e}")
        return image_path


def send_otp_email(receiver_email, otp):
    """Send registration OTP."""
    try:
        msg = EmailMessage()
        msg['Subject'] = "Your OTP — Pixhare"
        msg['From'] = Config.EMAIL_USER
        msg['To'] = receiver_email.strip()

        msg.set_content(f"""Hello,

Your OTP for Pixhare registration is: {otp}

This OTP is valid for 5 minutes. Do not share it with anyone.

Thanks,
Pixhare Team
""")

        print(f"🔑 Sending OTP to {receiver_email}")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(Config.EMAIL_USER, Config.EMAIL_PASS)
            smtp.send_message(msg)

        print(f"✅ OTP email sent to {receiver_email}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Gmail Authentication Error: {e}")
        return False

    except Exception as e:
        print(f"❌ OTP Email Error: {e}")
        return False


def send_reset_otp_email(receiver_email, otp):
    """Send password reset OTP."""
    try:
        msg = EmailMessage()
        msg['Subject'] = "Reset Your Pixhare Password"
        msg['From'] = Config.EMAIL_USER
        msg['To'] = receiver_email.strip()

        msg.set_content(f"""Hello,

We received a request to reset your Pixhare password.

Your OTP is: {otp}

This OTP is valid for 5 minutes.

Thanks,
Pixhare Team
""")

        print(f"🔑 Sending reset OTP to {receiver_email}")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(Config.EMAIL_USER, Config.EMAIL_PASS)
            smtp.send_message(msg)

        print(f"✅ Reset OTP email sent to {receiver_email}")
        return True

    except Exception as e:
        print(f"❌ Reset OTP email error: {e}")
        return False


def send_gallery_email(receiver_email, gallery_link, guest_name):
    """Send the personalised gallery link to a guest."""
    try:
        msg = EmailMessage()
        msg['Subject'] = "Your Event Photo Gallery - Pixhare"
        msg['From'] = Config.EMAIL_USER
        msg['To'] = receiver_email

        msg.set_content(f"""Hi {guest_name},

Thanks for attending the event!

Here is your private photo gallery:
{gallery_link}

Enjoy your photos!
— Pixhare
""")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(Config.EMAIL_USER, Config.EMAIL_PASS)
            smtp.send_message(msg)

        print(f"✅ Gallery email sent to {receiver_email}")

    except Exception as e:
        print(f"❌ Error sending gallery email: {e}")
        

def match_and_send_for_event(event_name, app_context):
    """
    Production-grade matching pipeline:
    1. RetinaFace  — detect & align faces in event photos
    2. ArcFace     — generate 512-dim face embeddings
    3. FAISS       — lightning-fast similarity search (replaces slow cosine loop)
    4. Per-guest   — query FAISS index with selfie embedding → get matched photos
    5. Email       — send gallery link to each matched guest
    """
    import numpy as np
    import faiss
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from deepface import DeepFace

    MODEL     = "ArcFace"
    BACKEND   = "retinaface"
    THRESHOLD = 0.40          # cosine distance threshold (lower = stricter)
    DIM       = 512           # ArcFace embedding dimension

    with app_context:
        guests = Guest.query.filter_by(event_name=event_name).all()
        event_photos_folder = os.path.join('static', 'photos', event_name)

        if not os.path.exists(event_photos_folder):
            print(f"[match] No photo folder for event: {event_name}")
            return

        photo_files = [f for f in os.listdir(event_photos_folder) if allowed_file(f)]

        if not photo_files:
            print(f"[match] No photos in folder: {event_name}")
            return

        if not guests:
            print(f"[match] No guests for event: {event_name}")
            return

        # ── STEP 1 & 2: RetinaFace detect + ArcFace embed all event photos ──
        print(f"[match] Embedding {len(photo_files)} photos with ArcFace + RetinaFace...")
        photo_names  = []   # ordered list of photo filenames
        embeddings   = []   # corresponding embeddings

        for photo in photo_files:
            photo_path = os.path.join(event_photos_folder, photo)
            try:
                emb = DeepFace.represent(
                    img_path         = photo_path,
                    model_name       = MODEL,
                    detector_backend = BACKEND,
                    enforce_detection= False
                )
                if emb:
                    vec = np.array(emb[0]['embedding'], dtype='float32')
                    # L2-normalise for cosine similarity via inner product
                    vec = vec / (np.linalg.norm(vec) + 1e-10)
                    photo_names.append(photo)
                    embeddings.append(vec)
            except Exception as e:
                print(f"[match] ⚠️ Could not embed {photo}: {e}")

        if not embeddings:
            print("[match] No valid photo embeddings. Aborting.")
            return

        # ── STEP 3: Build FAISS index ──
        print(f"[match] Building FAISS index for {len(embeddings)} photos...")
        matrix = np.stack(embeddings)   # shape: (n_photos, 512)

        # IndexFlatIP = exact inner product search
        # Since vecs are L2-normalised, inner product == cosine similarity
        index = faiss.IndexFlatIP(DIM)
        index.add(matrix)
        print(f"[match] FAISS index ready — {index.ntotal} vectors indexed.")

        # ── STEP 4: Match each guest selfie against FAISS index ──
        def process_guest(guest):
            match_folder = os.path.join('static', 'matches', guest.gallery_token)
            os.makedirs(match_folder, exist_ok=True)
            matched = 0

            try:
                selfie_raw = DeepFace.represent(
                    img_path         = guest.selfie_path,
                    model_name       = MODEL,
                    detector_backend = BACKEND,
                    enforce_detection= False
                )
                if not selfie_raw:
                    print(f"[match] ⚠️ No face in selfie for {guest.name}")
                    return guest, 0

                selfie_vec = np.array(selfie_raw[0]['embedding'], dtype='float32')
                selfie_vec = selfie_vec / (np.linalg.norm(selfie_vec) + 1e-10)
                selfie_vec = selfie_vec.reshape(1, -1)

                # FAISS search — returns cosine similarities (higher = more similar)
                scores, indices = index.search(selfie_vec, len(photo_names))

                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:
                        continue
                    cosine_distance = 1.0 - float(score)  # convert similarity → distance
                    if cosine_distance <= THRESHOLD:
                        photo     = photo_names[idx]
                        src_path  = os.path.join(event_photos_folder, photo)
                        dest_path = os.path.join(match_folder, photo)
                        if not os.path.exists(dest_path):
                            shutil.copy(src_path, dest_path)
                            print(f"[match] ✅ {photo} → {guest.name} (dist={cosine_distance:.3f})")
                        matched += 1

            except Exception as e:
                print(f"[match] ⚠️ Error for {guest.name}: {e}")

            return guest, matched

        # ── STEP 5: Process guests in parallel + email galleries ──
        print(f"[match] Matching {len(guests)} guest(s) in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_guest, g): g for g in guests}
            for future in as_completed(futures):
                try:
                    guest, count = future.result()
                    gallery_link = f"{BASE_URL}/gallery/{guest.gallery_token}"
                    send_gallery_email(guest.email, gallery_link, guest.name)
                    print(f"[match] 📧 Gallery sent to {guest.name} ({count} photos matched)")
                except Exception as e:
                    print(f"[match] ⚠️ Error processing guest: {e}")

        print(f"[match] ✅ Done for event: {event_name}")


# ─────────────────────────────────────────────
# Routes – Public
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')


# ─────────────────────────────────────────────
# Routes – Photographer Auth
# ─────────────────────────────────────────────
@app.route('/photographer/register', methods=['GET', 'POST'])
def photographer_register():
    if request.method == 'POST':
        name         = request.form['name'].strip()
        studio_name  = request.form['studio_name'].strip()
        email        = request.form['email'].strip()
        password     = request.form['password']

        if Photographer.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for('photographer_register'))

        otp = generate_otp()

        # Store all registration info + OTP expiry in session
        session['register_name']        = name
        session['register_studio_name'] = studio_name
        session['register_email']       = email
        session['register_password']    = password
        session['register_otp']         = otp
        session['otp_expiry']           = (datetime.now() + timedelta(minutes=5)).isoformat()

        sent = send_otp_email(email, otp)
        if sent:
            flash("OTP sent to your email. It expires in 5 minutes.", "info")
        else:
            flash("Could not send email — check terminal for the OTP (dev mode).", "warning")
        return redirect(url_for('verify_otp'))

    return render_template('photographer_register.html')


@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    email        = session.get('register_email')
    otp_sent     = session.get('register_otp')
    raw_password = session.get('register_password')
    otp_expiry   = session.get('otp_expiry')
    name         = session.get('register_name')
    studio_name  = session.get('register_studio_name')

    if not email or not otp_sent:
        flash("Session expired. Please register again.", "danger")
        return redirect(url_for('photographer_register'))

    # Calculate remaining seconds to pass to template
    remaining_seconds = 300  # default 5 min
    if otp_expiry:
        diff = datetime.fromisoformat(otp_expiry) - datetime.now()
        remaining_seconds = max(0, int(diff.total_seconds()))

    if request.method == 'POST':
        # ── Expiry check ──
        if remaining_seconds <= 0:
            flash("OTP has expired. Please register again.", "danger")
            for key in ('register_name', 'register_studio_name', 'register_email',
                        'register_password', 'register_otp', 'otp_expiry'):
                session.pop(key, None)
            return redirect(url_for('photographer_register'))

        entered_otp = request.form['otp']
        if entered_otp == otp_sent:
            hashed_pw = generate_password_hash(raw_password)
            new_photographer = Photographer(
                name        = name,
                studio_name = studio_name,
                email       = email,
                password    = hashed_pw,
                otp         = otp_sent,
                is_verified = True
            )
            db.session.add(new_photographer)
            db.session.commit()

            for key in ('register_name', 'register_studio_name', 'register_email',
                        'register_password', 'register_otp', 'otp_expiry'):
                session.pop(key, None)

            flash("Registration successful! Please login.", "success")
            return redirect(url_for('photographer_login'))
        else:
            flash("Invalid OTP. Please try again.", "danger")

    return render_template('otp_verify.html', email=email,
                           remaining_seconds=remaining_seconds)


@app.route('/photographer/login', methods=['GET', 'POST'])
def photographer_login():
    if request.method == 'POST':
        email    = request.form['email'].strip()
        password = request.form['password']
        photographer = Photographer.query.filter_by(email=email).first()

        if photographer and check_password_hash(photographer.password, password):
            session['email']       = email
            session['name']        = photographer.name
            session['studio_name'] = photographer.studio_name
            flash("Login successful!", "success")
            return redirect(url_for('photographer_dashboard'))
        else:
            flash("Invalid email or password.", "danger")

    return render_template('photographer_login.html')


# ─────────────────────────────────────────────
# Routes – Forgot Password
# ─────────────────────────────────────────────

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Step 1 — Enter email, send OTP."""
    if request.method == 'POST':
        email = request.form['email'].strip()
        photographer = Photographer.query.filter_by(email=email).first()

        if not photographer:
            flash("No account found with that email.", "danger")
            return redirect(url_for('forgot_password'))

        otp = generate_otp()
        session['fp_email']   = email
        session['fp_otp']     = otp
        session['fp_expiry']  = (datetime.now() + timedelta(minutes=5)).isoformat()

        sent = send_reset_otp_email(email, otp)
        if sent:
            flash("Password reset OTP sent to your email. It expires in 5 minutes.", "info")
        else:
            flash("Could not send email — check terminal for OTP (dev mode).", "warning")

        return redirect(url_for('forgot_password_verify'))

    return render_template('forgot_password.html')


@app.route('/forgot-password/verify', methods=['GET', 'POST'])
def forgot_password_verify():
    """Step 2 — Verify OTP."""
    email      = session.get('fp_email')
    otp_sent   = session.get('fp_otp')
    fp_expiry  = session.get('fp_expiry')

    if not email or not otp_sent:
        flash("Session expired. Please try again.", "danger")
        return redirect(url_for('forgot_password'))

    # Calculate remaining seconds for timer
    remaining_seconds = 300
    if fp_expiry:
        diff = datetime.fromisoformat(fp_expiry) - datetime.now()
        remaining_seconds = max(0, int(diff.total_seconds()))

    if request.method == 'POST':
        if remaining_seconds <= 0:
            flash("OTP has expired. Please request a new one.", "danger")
            session.pop('fp_email',  None)
            session.pop('fp_otp',    None)
            session.pop('fp_expiry', None)
            return redirect(url_for('forgot_password'))

        entered_otp = request.form['otp']
        if entered_otp == otp_sent:
            # OTP verified — allow password reset
            session['fp_verified'] = True
            session.pop('fp_otp',    None)
            session.pop('fp_expiry', None)
            return redirect(url_for('reset_password'))
        else:
            flash("Invalid OTP. Please try again.", "danger")

    return render_template('forgot_password_verify.html',
                           email=email,
                           remaining_seconds=remaining_seconds)


@app.route('/forgot-password/reset', methods=['GET', 'POST'])
def reset_password():
    """Step 3 — Set new password."""
    email       = session.get('fp_email')
    fp_verified = session.get('fp_verified')

    if not email or not fp_verified:
        flash("Unauthorised. Please start again.", "danger")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password  = request.form['password']
        confirm_password = request.form['confirm_password']

        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('reset_password'))

        if len(new_password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for('reset_password'))

        photographer = Photographer.query.filter_by(email=email).first()
        if photographer:
            photographer.password = generate_password_hash(new_password)
            db.session.commit()

        session.pop('fp_email',    None)
        session.pop('fp_verified', None)

        flash("Password reset successfully! Please login.", "success")
        return redirect(url_for('photographer_login'))

    return render_template('reset_password.html', email=email)


@app.route('/photographer/logout')
def photographer_logout():
    session.pop('email',       None)
    session.pop('name',        None)
    session.pop('studio_name', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('photographer_login'))


# ─────────────────────────────────────────────
# Routes – Photographer Dashboard & Events
# ─────────────────────────────────────────────
@app.route('/photographer/dashboard')
def photographer_dashboard():
    if 'email' not in session:
        return redirect(url_for('photographer_login'))

    photographer = Photographer.query.filter_by(email=session['email']).first()
    if not photographer:
        flash("User not found.", "danger")
        return redirect(url_for('photographer_login'))

    events = Event.query.filter_by(photographer_id=photographer.id).all()

    # Build stats dict for each event: guest count + photo count
    event_stats = {}
    for event in events:
        guest_count = Guest.query.filter_by(event_name=event.name).count()
        photo_folder = os.path.join('static', 'photos', event.name)
        photo_count = 0
        if os.path.exists(photo_folder):
            photo_count = len([
                f for f in os.listdir(photo_folder)
                if allowed_file(f)
            ])
        event_stats[event.name] = {
            'guests': guest_count,
            'photos': photo_count,
        }

    return render_template('photographer_dashboard.html',
                           events=events, event_stats=event_stats)


@app.route('/photographer/create_event', methods=['POST'])
def create_event():
    if 'email' not in session:
        return redirect(url_for('photographer_login'))

    event_name = request.form['event_name'].strip()
    event_date = request.form['event_date']

    photographer = Photographer.query.filter_by(email=session['email']).first()
    if not photographer:
        flash("User not found.", "danger")
        return redirect(url_for('photographer_login'))

    if Event.query.filter_by(name=event_name, photographer_id=photographer.id).first():
        flash('You already created an event with this name.', 'danger')
        return redirect(url_for('photographer_dashboard'))

    # Generate QR code pointing to the dynamic BASE_URL
    qr_filename = secure_filename(f"{event_name}_qr.png")
    qr_path     = os.path.join('static', 'qrcodes', qr_filename)
    os.makedirs(os.path.dirname(qr_path), exist_ok=True)
    generate_qr_code(f"{BASE_URL}/event/{event_name}/register", qr_path)

    new_event = Event(
        name            = event_name,
        date            = event_date,
        qr_filename     = qr_filename,
        photographer_id = photographer.id
    )
    db.session.add(new_event)
    db.session.commit()

    flash("Event created successfully!", "success")
    return redirect(url_for('photographer_dashboard'))


@app.route('/delete_event/<event_name>', methods=['POST'])
def delete_event(event_name):
    if 'email' not in session:
        return redirect(url_for('photographer_login'))

    # Remove all match folders for guests of this event
    guests = Guest.query.filter_by(event_name=event_name).all()
    for guest in guests:
        shutil.rmtree(os.path.join('static', 'matches', guest.gallery_token), ignore_errors=True)

    # Remove photo and selfie folders
    shutil.rmtree(os.path.join('static', 'photos',  event_name), ignore_errors=True)
    shutil.rmtree(os.path.join('static', 'guests',  event_name), ignore_errors=True)

    # Remove QR code
    event = Event.query.filter_by(name=event_name).first()
    if event and event.qr_filename:
        qr_path = os.path.join('static', 'qrcodes', event.qr_filename)
        if os.path.exists(qr_path):
            os.remove(qr_path)

    Guest.query.filter_by(event_name=event_name).delete()
    Event.query.filter_by(name=event_name).delete()
    db.session.commit()

    flash(f"Event '{event_name}' deleted successfully.", "success")
    return redirect(url_for('photographer_dashboard'))


@app.route('/event/<event_name>/delete_photos', methods=['POST'])
def delete_photos(event_name):
    """Delete selected photos from an event folder."""
    if 'email' not in session:
        return redirect(url_for('photographer_login'))

    filenames = request.form.getlist('photos_to_delete')
    folder    = os.path.join('static', 'photos', event_name)
    deleted   = 0

    for filename in filenames:
        # Sanitise — only allow the filename, no path traversal
        safe_name = os.path.basename(secure_filename(filename))
        file_path = os.path.join(folder, safe_name)
        if os.path.exists(file_path) and allowed_file(safe_name):
            os.remove(file_path)
            deleted += 1

    flash(f"{deleted} photo{'s' if deleted != 1 else ''} deleted successfully.", "success")
    return redirect(url_for('view_event_photos', event_name=event_name))



# ─────────────────────────────────────────────
# Routes – Guest Registration
# ─────────────────────────────────────────────
@app.route('/event/<event_name>/register', methods=['GET', 'POST'])
def guest_register(event_name):
    # ── Validate event exists before showing the form ──
    event = Event.query.filter_by(name=event_name).first()
    if not event:
        return render_template('event_not_found.html', event_name=event_name), 404

    if request.method == 'POST':
        name   = request.form['name'].strip()
        email  = request.form['email'].strip()
        selfie = request.files.get('selfie')

        # Validate selfie upload
        if not selfie or not allowed_file(selfie.filename):
            flash("Please upload a valid JPG or PNG selfie.", "danger")
            return redirect(request.url)

        selfie_folder   = os.path.join('static', 'guests', event_name)
        os.makedirs(selfie_folder, exist_ok=True)

        selfie_filename = secure_filename(f"{uuid4()}.jpg")
        selfie_path     = os.path.join(selfie_folder, selfie_filename)
        selfie.save(selfie_path)

        gallery_token = str(uuid4())
        guest = Guest(
            name          = name,
            email         = email,
            event_name    = event_name,
            selfie_path   = selfie_path,
            gallery_token = gallery_token
        )
        db.session.add(guest)
        db.session.commit()

        return render_template('guest_success.html', guest_name=name)

    return render_template('guest_register.html', event_name=event_name)


# ─────────────────────────────────────────────
# Routes – Photo Upload
# ─────────────────────────────────────────────
@app.route('/upload/<event_name>', methods=['GET', 'POST'])
def upload_photos(event_name):
    if 'email' not in session:
        return redirect(url_for('photographer_login'))

    if request.method == 'POST':
        files         = request.files.getlist('photos')
        upload_folder = os.path.join('static', 'photos', event_name)
        os.makedirs(upload_folder, exist_ok=True)

        saved_count   = 0
        skipped_count = 0

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(upload_folder, filename))
                saved_count += 1
            else:
                skipped_count += 1

        flash(
            f"{saved_count} photo(s) uploaded."
            + (f" {skipped_count} file(s) skipped (invalid type)." if skipped_count else ""),
            "success" if saved_count else "warning"
        )
        return render_template('upload_photos.html', event_name=event_name, uploaded=True)

    return render_template('upload_photos.html', event_name=event_name, uploaded=False)


# ─────────────────────────────────────────────
# Routes – View Photos & Trigger Matching
# ─────────────────────────────────────────────
@app.route('/event/<event_name>/photos', methods=['GET', 'POST'])
def view_event_photos(event_name):
    if 'email' not in session:
        return redirect(url_for('photographer_login'))

    folder = os.path.join('static', 'photos', event_name)
    if not os.path.exists(folder):
        return f"No photos found for event: {event_name}", 404

    photos = [f for f in os.listdir(folder) if allowed_file(f)]

    if request.method == 'POST':
        # ── Run matching asynchronously so the page responds immediately ──
        thread = Thread(
            target=match_and_send_for_event,
            args=(event_name, app.app_context())
        )
        thread.start()

        flash(
            "Face matching started in the background. "
            "Gallery links will be emailed to guests once complete.",
            "info"
        )
        return redirect(url_for('view_event_photos', event_name=event_name))

    return render_template('view_photos.html', event_name=event_name, photos=photos)


# ─────────────────────────────────────────────
# Routes – Send Gallery (Dashboard Button)
# ─────────────────────────────────────────────
@app.route('/send_gallery/<event_name>', methods=['POST'])
def send_gallery(event_name):
    if 'email' not in session:
        return redirect(url_for('photographer_login'))

    if not os.path.exists(os.path.join('static', 'photos', event_name)):
        flash(f"No event photos found for '{event_name}'.", "danger")
        return redirect(url_for('photographer_dashboard'))

    thread = Thread(
        target=match_and_send_for_event,
        args=(event_name, app.app_context())
    )
    thread.start()

    flash(
        f"Matching started for '{event_name}'. "
        "Gallery emails will be sent to guests automatically.",
        "info"
    )
    return redirect(url_for('photographer_dashboard'))


# ─────────────────────────────────────────────
# Routes – Guest Gallery
# ─────────────────────────────────────────────
@app.route('/gallery/<uuid>')
def view_gallery(uuid):
    guest  = Guest.query.filter_by(gallery_token=uuid).first_or_404()
    folder = os.path.join('static', 'matches', uuid)

    if not os.path.exists(folder) or not os.listdir(folder):
        return render_template('gallery.html', guest=guest, photos=[], message="No matched photos yet. Please check back soon.")

    photos = [f"/static/matches/{uuid}/{f}" for f in os.listdir(folder) if allowed_file(f)]
    return render_template('gallery.html', guest=guest, photos=photos)


# ─────────────────────────────────────────────
# Routes – Chatbot
# ─────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    """RAG chatbot endpoint with multilingual support."""
    data     = request.get_json()
    query    = (data.get('message',  '') or '').strip()
    language = (data.get('language', 'English') or 'English').strip()

    if not query:
        return jsonify({'reply': 'Please type a message.'})

    history = data.get('history', [])

    try:
        reply = get_answer(query, history, language)
    except Exception as e:
        print(f"[chat] Error: {e}")
        reply = "I'm having trouble right now. Please try again in a moment."

    return jsonify({'reply': reply})


@app.route('/test-tts')
def test_tts_route():
    """Quick browser test for Edge TTS Tamil voice."""
    return '''
<!DOCTYPE html>
<html>
<head><title>TTS Test</title></head>
<body style="font-family:sans-serif;padding:30px;">
  <h2>Edge TTS Test</h2>
  <button onclick="testTTS()" style="padding:12px 24px;font-size:1rem;background:#0f7b56;color:white;border:none;border-radius:8px;cursor:pointer;">
    🔊 Test Tamil Voice
  </button>
  <div id="status" style="margin-top:16px;font-size:0.9rem;"></div>
  <script>
  function testTTS() {
    document.getElementById('status').textContent = 'Requesting audio...';
    fetch('/tts', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text:'வணக்கம், நான் பிக்ஸி', language:'Tamil'})
    })
    .then(function(r) {
      document.getElementById('status').textContent = 'Got response: ' + r.status + ' ' + r.headers.get('content-type');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.blob();
    })
    .then(function(blob) {
      document.getElementById('status').textContent = 'Got blob: ' + blob.size + ' bytes, type: ' + blob.type;
      var url = URL.createObjectURL(blob);
      var audio = new Audio(url);
      audio.play().then(function() {
        document.getElementById('status').textContent += ' ✅ Playing!';
      }).catch(function(e) {
        document.getElementById('status').textContent += ' ❌ Play failed: ' + e.message;
      });
    })
    .catch(function(e) {
      document.getElementById('status').textContent = '❌ Error: ' + e.message;
    });
  }
  </script>
</body>
</html>
'''


@app.route('/tts', methods=['POST'])
def tts():
    """Text-to-speech using Microsoft Edge TTS."""
    import asyncio, edge_tts, io, re

    data = request.get_json()
    text = (data.get('text', '') or '').strip()
    lang = (data.get('language', 'English') or 'English').strip()

    if not text:
        return jsonify({'error': 'No text'}), 400

    VOICE_MAP = {
        'Tamil':'ta-IN-PallaviNeural',    'Hindi':'hi-IN-SwaraNeural',
        'Telugu':'te-IN-ShrutiNeural',    'Malayalam':'ml-IN-SobhanaNeural',
        'Kannada':'kn-IN-GaganNeural',    'Bengali':'bn-IN-TanishaaNeural',
        'Marathi':'mr-IN-AarohiNeural',   'Spanish':'es-ES-ElviraNeural',
        'French':'fr-FR-DeniseNeural',    'Arabic':'ar-SA-ZariyahNeural',
        'Japanese':'ja-JP-NanamiNeural',  'Chinese':'zh-CN-XiaoxiaoNeural',
        'German':'de-DE-KatjaNeural',     'English':'en-US-AriaNeural',
    }
    voice = VOICE_MAP.get(lang, 'en-US-AriaNeural')

    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    clean = re.sub(r'\*(.*?)\*',     r'\1', clean)
    clean = re.sub(r'<[^>]+>',       ' ',   clean)
    clean = re.sub(r'^\s*\d+[\.\)]\s*', '', clean, flags=re.MULTILINE)
    clean = re.sub(r'[•\-\*#]',      ' ',   clean)
    clean = re.sub(r'\s+',           ' ',   clean).strip()[:400]

    if not clean:
        return jsonify({'error': 'Empty text'}), 400

    print(f"[TTS] {lang} → {voice}: {clean[:60]}...")

    try:
        async def generate():
            buf = io.BytesIO()
            communicate = edge_tts.Communicate(clean, voice)
            async for chunk in communicate.stream():
                if chunk['type'] == 'audio':
                    buf.write(chunk['data'])
            return buf.getvalue()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_data = loop.run_until_complete(generate())
        loop.close()

        if not audio_data:
            return jsonify({'error': 'No audio'}), 500

        from flask import Response
        return Response(audio_data, mimetype='audio/mpeg',
                        headers={'Cache-Control': 'no-cache'})
    except Exception as e:
        print(f"[TTS] Error: {type(e).__name__}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio using Groq Whisper API — fast, free, no local model."""
    import tempfile

    audio_file = request.files.get('audio')
    language   = request.form.get('language', 'English').strip()

    if not audio_file:
        return jsonify({'text': '', 'error': 'No audio file received'})

    LANG_MAP = {
        'English':'en', 'Tamil':'ta', 'Hindi':'hi',
        'Telugu':'te', 'Malayalam':'ml', 'Kannada':'kn',
        'Bengali':'bn', 'Marathi':'mr', 'Spanish':'es',
        'French':'fr', 'Arabic':'ar', 'Japanese':'ja',
        'Chinese':'zh', 'German':'de'
    }
    whisper_lang = LANG_MAP.get(language, 'en')
    tmp_path = None

    try:
        # Read audio bytes
        audio_bytes = audio_file.read()
        print(f"[Whisper] Audio size: {len(audio_bytes)} bytes, lang={whisper_lang}")

        if len(audio_bytes) < 1000:
            print("[Whisper] Audio too small — likely no speech recorded")
            return jsonify({'text': '', 'error': 'Audio too short'})

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        print(f"[Whisper] Sending to Groq whisper-large-v3-turbo...")

        from groq import Groq as GroqClient
        client = GroqClient(api_key=Config.GROQ_API_KEY)

        with open(tmp_path, 'rb') as f:
            result = client.audio.transcriptions.create(
                file            = ('audio.webm', f, 'audio/webm'),
                model           = "whisper-large-v3-turbo",
                language        = whisper_lang,
                response_format = "text",
            )

        text = (result or "").strip()
        print(f"[Whisper] ✅ Result ({language}): '{text}'")
        return jsonify({'text': text})

    except Exception as e:
        print(f"[Whisper] ❌ Error: {type(e).__name__}: {e}")
        return jsonify({'text': '', 'error': str(e)})

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    Thread(target=init_chatbot, daemon=True).start()
    # debug=True only in local dev, False in production
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))