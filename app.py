from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from models import db, Photographer, Guest, Event, EventPhoto, PhotoFaceEmbedding
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
from threading import Thread
from email.message import EmailMessage
from uuid import uuid4
from datetime import datetime, timedelta
import random, smtplib, qrcode, os, shutil, cv2, io, json, base64, tempfile
import numpy as np
from PIL import Image
from chatbot import get_answer, initialize as init_chatbot
import storage


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

# Supabase (and most hosted Postgres) close idle connections after a while.
# Without this, Flask-SQLAlchemy's pool can hand out a connection the
# server already dropped, causing "server closed the connection
# unexpectedly" on the next query after any period of inactivity — exactly
# what happens between manual test requests on a dev server.
#   pool_pre_ping: cheap "is this connection still alive" check before each
#                  use; transparently reconnects if it's dead.
#   pool_recycle:  proactively replace connections older than this many
#                  seconds, before the server has a chance to drop them.
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280,
}

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


def generate_qr_code_bytes(data):
    """Generate a QR code with the Pixhare mark centered in it, returned as
    PNG bytes (no local disk). Uses HIGH error correction (~30% of the code
    can be obscured and still scan) specifically so the center logo doesn't
    break scannability — the logo is sized well under that recoverable
    fraction, with a white backdrop for clean contrast against the modules
    directly behind it."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=5
    )
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color='black', back_color='white').convert('RGB')

    try:
        logo_path = os.path.join(app.root_path, 'static', 'images', 'logo.png')
        logo = Image.open(logo_path).convert('RGB')
        qr_w, qr_h = qr_img.size

        # ~20% of the QR width — a clear brand mark, small enough that
        # ERROR_CORRECT_H can still recover the modules it covers.
        logo_size = int(qr_w * 0.20)
        logo = logo.resize((logo_size, logo_size), Image.LANCZOS)

        # White backdrop, slightly larger than the logo, so it reads
        # cleanly regardless of which modules sit behind it.
        pad = int(logo_size * 0.12)
        backdrop_size = logo_size + pad * 2
        backdrop = Image.new('RGB', (backdrop_size, backdrop_size), 'white')

        backdrop_pos = ((qr_w - backdrop_size) // 2, (qr_h - backdrop_size) // 2)
        qr_img.paste(backdrop, backdrop_pos)
        logo_pos = ((qr_w - logo_size) // 2, (qr_h - logo_size) // 2)
        qr_img.paste(logo, logo_pos)
    except Exception as e:
        print(f"[qr] ⚠️ Could not add logo to QR code: {e}")

    buf = io.BytesIO()
    qr_img.save(buf, format='PNG')
    return buf.getvalue()


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
        print(f"❌ Gmail SMTP Authentication Error: {e}")
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
    """Send the personalised gallery link to a guest — first time only."""
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


def send_new_photos_email(receiver_email, gallery_link, guest_name, new_count):
    """Notify a guest who already has a gallery that more photos were just added."""
    try:
        msg = EmailMessage()
        msg['Subject'] = "New photos added to your gallery - Pixhare"
        msg['From'] = Config.EMAIL_USER
        msg['To'] = receiver_email

        msg.set_content(f"""Hi {guest_name},

{new_count} new photo{'s' if new_count != 1 else ''} of you just got added to your gallery:
{gallery_link}

Enjoy!
— Pixhare
""")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(Config.EMAIL_USER, Config.EMAIL_PASS)
            smtp.send_message(msg)

        print(f"✅ 'New photos' email sent to {receiver_email}")

    except Exception as e:
        print(f"❌ Error sending 'new photos' email: {e}")
        

def match_and_send_for_event(event_name, app_context):
    """
    Incremental matching pipeline — only processes photos uploaded since the
    last run (tracked via EventPhoto.matched), so a repeat 'Send Gallery'
    click after more photos are added doesn't re-embed the whole event.

    1. RetinaFace  — detect & align every face in each NEW event photo
    2. ArcFace     — generate 512-dim face embeddings
    3. FAISS       — similarity search over just the new photos
    4. Per-guest   — query with all captured angles → get newly matched photos,
                     copied ADD-ON-TOP of whatever's already in their gallery
    5. Email       — first-time guests get the full "gallery ready" email;
                     returning guests only get emailed if new photos matched
    """
    import numpy as np
    import faiss
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from deepface import DeepFace

    MODEL     = "ArcFace"
    BACKEND   = "retinaface"
    THRESHOLD = 0.40          # cosine distance threshold (lower = stricter)
    DIM       = 512           # ArcFace embedding dimension

    # Quality gate: RetinaFace occasionally "detects" a face in a shirt
    # pattern, a blurred background object, etc. Below this confidence,
    # the detection is more likely junk than a real face — skip it rather
    # than let a spurious low-quality embedding pollute the index.
    MIN_FACE_CONFIDENCE = 0.85

    # Group/crowd photos often contain faces well under 100px wide. ArcFace
    # embeds whatever crop RetinaFace hands it — a tiny, low-res crop makes
    # for a noisier embedding. For small faces we additionally upscale the
    # crop and re-embed it, adding a SECOND candidate embedding for that
    # face. Both compete for the same photo in matching (best distance
    # wins), so this only ever helps, never overrides the original.
    MIN_FACE_SIZE_FOR_REFINEMENT = 100    # px, bounding box width
    REFINEMENT_UPSCALE_TARGET    = 220    # px, upscale small crops to at least this

    # Soft reranking: a borderline-distance match from a high-confidence,
    # cleanly detected face should be trusted slightly more than the same
    # distance from a barely-passing detection. This nudges (never flips a
    # clearly-wrong match into an accept) the effective distance used for
    # the threshold decision, bounded to a small range.
    QUALITY_BONUS_SCALE = 0.05

    with app_context:
        guests = Guest.query.filter_by(event_name=event_name).all()
        if not guests:
            print(f"[match] No guests for event: {event_name}")
            return

        # Photos uploaded since the last run — the expensive embedding step
        # only ever runs on these.
        new_photo_rows = EventPhoto.query.filter_by(event_name=event_name, matched=False).all()
        new_filenames = [row.filename for row in new_photo_rows if allowed_file(row.filename)]

        # Guests who have never been matched before need checking against
        # the event's FULL photo history, not just this run's new batch —
        # otherwise someone who registers after photos are already uploaded
        # would never be matched to anything that came before them.
        never_matched_guests = [g for g in guests if g.gallery_sent_at is None]

        if not new_filenames and not never_matched_guests:
            print(f"[match] Nothing new for event: {event_name} — no new photos, no new guests")
            return

        def load_image_array(storage_path):
            """Download a Storage object and decode it into a BGR numpy array
            (what DeepFace/cv2 expect) — no local disk involved."""
            data = storage.download_bytes(storage_path)
            if data is None:
                return None
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Each phase window from guest_register.html's guided recording
        # script (0-1.7s center, 1.7-3.35s right, 3.35-5s left). Rather than
        # trusting one fixed timestamp — which might catch the guest mid-turn
        # or before they've reacted to the prompt — sample a few candidates
        # from the LATTER part of each window (giving reaction + turn time)
        # and keep whichever has the highest face-detection confidence.
        VIDEO_PHASE_CANDIDATES = {
            'center': [1.00, 1.30, 1.55],
            'right':  [2.30, 2.70, 3.05],
            'left':   [3.90, 4.30, 4.65],
        }

        def extract_best_video_frames(storage_path):
            """Download a guest's selfie video and, for each guided phase,
            pick the clearest candidate frame (highest RetinaFace detection
            confidence) rather than one blind fixed timestamp. Returns a
            list of up to 3 BGR numpy arrays, one per phase that found an
            acceptable face. cv2.VideoCapture needs a real file path (no
            in-memory buffer support), so this writes to a short-lived temp
            file and always cleans it up, even on failure."""
            data = storage.download_bytes(storage_path)
            if data is None:
                return []

            tmp_path = os.path.join(tempfile.gettempdir(), f"pixhare_video_{uuid4().hex}.webm")
            best_frames = []
            try:
                with open(tmp_path, 'wb') as f:
                    f.write(data)

                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    print(f"[match] ⚠️ Could not open video {storage_path}")
                    return []

                for phase, timestamps in VIDEO_PHASE_CANDIDATES.items():
                    best_conf  = -1.0
                    best_frame = None
                    for ts in timestamps:
                        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            continue
                        try:
                            raw = DeepFace.represent(
                                img_path         = frame,
                                model_name       = MODEL,
                                detector_backend = BACKEND,
                                enforce_detection= False
                            )
                            if not raw:
                                continue
                            conf = max(f.get('face_confidence', 0.0) for f in raw)
                            if conf > best_conf:
                                best_conf  = conf
                                best_frame = frame
                        except Exception:
                            continue  # this candidate failed, try the next timestamp

                    if best_frame is not None and best_conf >= MIN_FACE_CONFIDENCE:
                        best_frames.append(best_frame)
                    else:
                        print(f"[match] ⚠️ No acceptable '{phase}' frame found for video {storage_path}")

                cap.release()
            except Exception as e:
                print(f"[match] ⚠️ Video frame extraction failed for {storage_path}: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            return best_frames

        def refine_small_face(img, facial_area):
            """Crop a small detected face out of the full photo, upscale it,
            and re-embed just that crop for a cleaner second embedding.
            Returns (vec, confidence) or None if it can't be refined."""
            try:
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                if w >= MIN_FACE_SIZE_FOR_REFINEMENT:
                    return None
                # Small margin around the box so alignment isn't cut off
                pad = int(max(w, h) * 0.25)
                y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    return None
                scale = max(1.0, REFINEMENT_UPSCALE_TARGET / max(w, h))
                upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                refined = DeepFace.represent(
                    img_path         = upscaled,
                    model_name       = MODEL,
                    detector_backend = BACKEND,
                    enforce_detection= False
                )
                if not refined:
                    return None
                best = max(refined, key=lambda f: f.get('face_confidence', 1.0))
                conf = best.get('face_confidence', 1.0)
                if conf < MIN_FACE_CONFIDENCE:
                    return None
                vec = np.array(best['embedding'], dtype='float32')
                vec = vec / (np.linalg.norm(vec) + 1e-10)
                return vec, conf
            except Exception:
                return None

        # ── STEP 1 & 2: RetinaFace detect + ArcFace embed the NEW photos ──
        # Every face in a photo gets embedded, not just one — a photo with
        # several people contributes one entry per person, plus a second
        # refined entry for any small/distant face. Each embedding is cached
        # to PhotoFaceEmbedding so it's never recomputed on a later run.
        new_names = []   # one entry per embedding (a photo/face can repeat)
        new_vecs  = []
        new_confs = []

        if new_filenames:
            print(f"[match] Embedding {len(new_filenames)} new photo(s) with ArcFace + RetinaFace...")
            skipped_low_confidence = 0
            refined_count = 0

            for photo in new_filenames:
                img = load_image_array(f"photos/{event_name}/{photo}")
                if img is None:
                    print(f"[match] ⚠️ Could not download {photo}")
                    continue
                try:
                    faces = DeepFace.represent(
                        img_path         = img,
                        model_name       = MODEL,
                        detector_backend = BACKEND,
                        enforce_detection= False
                    )
                    for face in faces or []:
                        confidence = face.get('face_confidence', 1.0)
                        if confidence < MIN_FACE_CONFIDENCE:
                            skipped_low_confidence += 1
                            continue
                        vec = np.array(face['embedding'], dtype='float32')
                        # L2-normalise for cosine similarity via inner product
                        vec = vec / (np.linalg.norm(vec) + 1e-10)
                        new_names.append(photo)
                        new_vecs.append(vec)
                        new_confs.append(confidence)
                        db.session.add(PhotoFaceEmbedding(
                            event_name = event_name,
                            filename   = photo,
                            embedding  = json.dumps(vec.tolist()),
                            confidence = confidence
                        ))

                        # Second candidate embedding for small/distant faces —
                        # common in group and crowd photos.
                        refined = refine_small_face(img, face.get('facial_area', {}))
                        if refined:
                            r_vec, r_conf = refined
                            new_names.append(photo)
                            new_vecs.append(r_vec)
                            new_confs.append(r_conf)
                            refined_count += 1
                            db.session.add(PhotoFaceEmbedding(
                                event_name = event_name,
                                filename   = photo,
                                embedding  = json.dumps(r_vec.tolist()),
                                confidence = r_conf
                            ))
                except Exception as e:
                    print(f"[match] ⚠️ Could not embed {photo}: {e}")

            if skipped_low_confidence:
                print(f"[match] Filtered {skipped_low_confidence} low-confidence face detection(s) (< {MIN_FACE_CONFIDENCE})")
            if refined_count:
                print(f"[match] Added {refined_count} refined embedding(s) for small/distant faces")

            # Mark every attempted photo as processed regardless of outcome —
            # a photo that failed to download/embed just won't contribute a
            # match; retrying it forever on every future run isn't useful.
            for row in new_photo_rows:
                row.matched = True
            db.session.commit()

        # ── STEP 3: Build the "new photos only" FAISS index ──
        # This is what already-matched (returning) guests are checked
        # against — cheap, since it's just this run's batch.
        new_index = None
        if new_vecs:
            new_index = faiss.IndexFlatIP(DIM)
            new_index.add(np.stack(new_vecs))
            print(f"[match] New-photos index ready — {new_index.ntotal} face(s).")

        # ── Build the "full history" FAISS index, only if needed ──
        # Only guests being matched for the very first time need this — skip
        # the DB read entirely when everyone present is a returning guest.
        full_index = None
        full_names = []
        full_confs = []
        if never_matched_guests:
            rows = PhotoFaceEmbedding.query.filter_by(event_name=event_name).all()
            for row in rows:
                full_names.append(row.filename)
                full_confs.append(row.confidence if row.confidence is not None else 1.0)
            if rows:
                full_matrix = np.array([json.loads(r.embedding) for r in rows], dtype='float32')
                full_index = faiss.IndexFlatIP(DIM)
                full_index.add(full_matrix)
                print(f"[match] Full-history index ready — {full_index.ntotal} face(s) — for {len(never_matched_guests)} new guest(s).")

        # ── STEP 4: Match each guest against the appropriate index ──
        # New photos get copied into the guest's existing match folder — this
        # ADDS to whatever they already have, nothing gets removed/replaced.
        def process_guest(guest):
            matched = 0
            is_first_run  = guest.gallery_sent_at is None
            index         = full_index if is_first_run else new_index
            photo_names   = full_names if is_first_run else new_names
            photo_confs   = full_confs if is_first_run else new_confs

            if index is None or index.ntotal == 0:
                return guest, 0

            try:
                if guest.selfie_is_video and guest.selfie_center_path:
                    frames = extract_best_video_frames(guest.selfie_center_path)
                    if not frames:
                        print(f"[match] ⚠️ Could not extract any frames from video for {guest.name}")
                        return guest, 0
                elif guest.selfie_center_path:
                    # Upload fallback — a single static photo, used as-is.
                    img = load_image_array(guest.selfie_center_path)
                    frames = [img] if img is not None else []
                else:
                    print(f"[match] ⚠️ No selfie on file for {guest.name}")
                    return guest, 0

                if not frames:
                    print(f"[match] ⚠️ No selfie on file for {guest.name}")
                    return guest, 0

                angle_vecs = []
                for img in frames:
                    try:
                        raw = DeepFace.represent(
                            img_path         = img,
                            model_name       = MODEL,
                            detector_backend = BACKEND,
                            enforce_detection= False
                        )
                        if raw:
                            best_face = max(raw, key=lambda f: f.get('face_confidence', 1.0))
                            if best_face.get('face_confidence', 1.0) < MIN_FACE_CONFIDENCE:
                                print(f"[match] ⚠️ Low-confidence selfie frame skipped for {guest.name}")
                                continue
                            v = np.array(best_face['embedding'], dtype='float32')
                            v = v / (np.linalg.norm(v) + 1e-10)
                            angle_vecs.append(v)
                    except Exception as e:
                        print(f"[match] ⚠️ Could not embed a selfie frame for {guest.name}: {e}")

                if not angle_vecs:
                    print(f"[match] ⚠️ No face detected in any selfie for {guest.name}")
                    return guest, 0

                # For each candidate photo, keep the BEST (lowest) QUALITY-
                # ADJUSTED distance seen across all of the guest's captured
                # angles. The adjustment gives a small edge to matches coming
                # from higher-confidence face detections — it can nudge a
                # borderline case, but is bounded small enough that it never
                # turns a clearly-wrong match into an accept. We still search
                # every photo per angle (no top-K cap).
                best_distance = {}
                for vec in angle_vecs:
                    vec = vec.reshape(1, -1)
                    scores, indices = index.search(vec, len(photo_names))
                    for score, idx in zip(scores[0], indices[0]):
                        if idx < 0:
                            continue
                        distance = 1.0 - float(score)
                        confidence = photo_confs[idx] if idx < len(photo_confs) else 1.0
                        quality_bonus = max(0.0, confidence - MIN_FACE_CONFIDENCE) * QUALITY_BONUS_SCALE
                        adjusted_distance = distance - quality_bonus
                        photo = photo_names[idx]
                        if photo not in best_distance or adjusted_distance < best_distance[photo]:
                            best_distance[photo] = adjusted_distance

                for photo, cosine_distance in best_distance.items():
                    if cosine_distance <= THRESHOLD:
                        try:
                            storage.copy_file(
                                f"photos/{event_name}/{photo}",
                                f"matches/{guest.gallery_token}/{photo}"
                            )
                            print(f"[match] ✅ {photo} → {guest.name} (dist={cosine_distance:.3f})")
                            matched += 1
                        except Exception as e:
                            print(f"[match] ⚠️ Could not copy {photo} for {guest.name}: {e}")

            except Exception as e:
                print(f"[match] ⚠️ Error for {guest.name}: {e}")

            return guest, matched

        # ── STEP 5: Process guests in parallel, then email on the main thread ──
        print(f"[match] Matching {len(guests)} guest(s) — {len(new_filenames)} new photo(s), {len(never_matched_guests)} first-time guest(s)...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_guest, g): g for g in guests}
            for future in as_completed(futures):
                try:
                    guest, count = future.result()
                    gallery_link = f"{BASE_URL}/gallery/{guest.gallery_token}"

                    if guest.gallery_sent_at is None:
                        # First time this guest has ever been matched for this
                        # event — send their permanent gallery link regardless
                        # of whether this batch matched them, so they have it
                        # ready for whenever new photos of them do show up.
                        send_gallery_email(guest.email, gallery_link, guest.name)
                        guest.gallery_sent_at = datetime.utcnow()
                        db.session.commit()
                        print(f"[match] 📧 Gallery link sent to {guest.name} ({count} photo(s) matched)")
                    elif count > 0:
                        # Returning guest — only notify if this run actually
                        # added something new to their existing gallery.
                        send_new_photos_email(guest.email, gallery_link, guest.name, count)
                        print(f"[match] 📧 'New photos' email sent to {guest.name} ({count} new)")
                    else:
                        print(f"[match] No new matches for {guest.name} this run — no email sent")

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
                is_verified = True,
                scan_token  = uuid4().hex
            )
            db.session.add(new_photographer)
            db.session.commit()

            # One QR per photographer, generated once — guests scan this
            # and get routed to whichever event is live today (see
            # /scan/<scan_token>), instead of a separate QR per event.
            try:
                qr_storage_path = f"qrcodes/{new_photographer.scan_token}.png"
                qr_bytes = generate_qr_code_bytes(f"{BASE_URL}/scan/{new_photographer.scan_token}")
                storage.upload_bytes(qr_storage_path, qr_bytes, content_type="image/png")
                new_photographer.qr_url = storage.get_public_url(qr_storage_path)
                db.session.commit()
            except Exception as e:
                print(f"[qr] ⚠️ Could not generate photographer QR at registration: {e}")

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

    # Backward compatibility — photographers who registered before the
    # one-QR-per-photographer change won't have a scan_token yet.
    if not photographer.scan_token:
        try:
            photographer.scan_token = uuid4().hex
            qr_storage_path = f"qrcodes/{photographer.scan_token}.png"
            qr_bytes = generate_qr_code_bytes(f"{BASE_URL}/scan/{photographer.scan_token}")
            storage.upload_bytes(qr_storage_path, qr_bytes, content_type="image/png")
            photographer.qr_url = storage.get_public_url(qr_storage_path)
            db.session.commit()
        except Exception as e:
            print(f"[qr] ⚠️ Could not lazily generate photographer QR: {e}")

    events = Event.query.filter_by(photographer_id=photographer.id).all()

    # Build stats dict for each event: guest count + photo count.
    # photo_count is now a cached column on Event (kept in sync by
    # upload_photos/delete_photos) instead of a Storage list() network
    # call per event — that was the real cause of the slow dashboard load.
    event_stats = {}
    for event in events:
        guest_count = Guest.query.filter_by(event_name=event.name).count()
        event_stats[event.name] = {
            'guests': guest_count,
            'photos': event.photo_count or 0,
        }

    # Base64-embed the QR for the poster canvas — loading it as a plain
    # <img src="https://...supabase.co/...">  with crossOrigin='anonymous'
    # depends on Supabase Storage returning CORS headers on that response,
    # which isn't reliable to depend on. A data: URI has no cross-origin
    # fetch at all, so canvas.toDataURL() (used for the poster download)
    # never fails with a tainted-canvas error.
    qr_data_uri = None
    if photographer.scan_token:
        qr_bytes = storage.download_bytes(f"qrcodes/{photographer.scan_token}.png")
        if qr_bytes:
            qr_data_uri = "data:image/png;base64," + base64.b64encode(qr_bytes).decode('ascii')

    return render_template('photographer_dashboard.html',
                           events=events, event_stats=event_stats,
                           photographer=photographer, qr_data_uri=qr_data_uri)


@app.route('/photographer/create_event', methods=['POST'])
def create_event():
    if 'email' not in session:
        return redirect(url_for('photographer_login'))

    event_name  = request.form['event_name'].strip()
    event_date  = request.form['event_date']
    venue       = request.form.get('venue', '').strip()
    event_time  = request.form.get('event_time', '').strip()

    photographer = Photographer.query.filter_by(email=session['email']).first()
    if not photographer:
        flash("User not found.", "danger")
        return redirect(url_for('photographer_login'))

    if Event.query.filter_by(name=event_name).first():
        flash('That event name is already taken. Event names must be unique across all photographers on Pixhare — try something more specific, like adding your studio name or the date.', 'danger')
        return redirect(url_for('photographer_dashboard'))

    # No per-event QR anymore — guests scan the photographer's single QR
    # (generated once at registration) and get routed to whichever event
    # is live today. See /scan/<scan_token>.
    new_event = Event(
        name            = event_name,
        date            = event_date,
        venue           = venue or None,
        event_time      = event_time or None,
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
        storage.delete_prefix(f"matches/{guest.gallery_token}")

    # Remove photo and selfie folders
    storage.delete_prefix(f"photos/{event_name}")
    storage.delete_prefix(f"guests/{event_name}")

    # No per-event QR to remove anymore — QR lives on the photographer now,
    # shared across all their events.

    Guest.query.filter_by(event_name=event_name).delete()
    EventPhoto.query.filter_by(event_name=event_name).delete()
    PhotoFaceEmbedding.query.filter_by(event_name=event_name).delete()
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
    deleted   = 0

    for filename in filenames:
        # Sanitise — only allow the filename, no path traversal
        safe_name = os.path.basename(secure_filename(filename))
        if allowed_file(safe_name):
            storage.delete_file(f"photos/{event_name}/{safe_name}")
            EventPhoto.query.filter_by(event_name=event_name, filename=safe_name).delete()
            deleted += 1

    if deleted:
        event = Event.query.filter_by(name=event_name).first()
        if event:
            event.photo_count = max(0, (event.photo_count or 0) - deleted)
    db.session.commit()

    flash(f"{deleted} photo{'s' if deleted != 1 else ''} deleted successfully.", "success")
    return redirect(url_for('view_event_photos', event_name=event_name))



# ─────────────────────────────────────────────
# Routes – Photographer QR scan (one QR per photographer)
# ─────────────────────────────────────────────
@app.route('/scan/<scan_token>')
def scan_photographer_qr(scan_token):
    photographer = Photographer.query.filter_by(scan_token=scan_token).first()
    if not photographer:
        return render_template('event_not_found.html', event_name=None), 404

    today = datetime.utcnow().strftime('%Y-%m-%d')
    live_events = Event.query.filter_by(
        photographer_id=photographer.id, date=today
    ).order_by(Event.event_time).all()

    if len(live_events) == 1:
        # Case 1 — exactly one live event today: go straight to it.
        # guest_register.html already shows event name/venue/date and the
        # "Take Selfie" capture flow, so no separate confirmation page.
        return redirect(url_for('guest_register', event_name=live_events[0].name))

    elif len(live_events) > 1:
        # Case 2 — multiple live events today: let the guest choose.
        return render_template('select_event.html',
                               events=live_events, photographer=photographer)

    else:
        # Case 3 — nothing live today.
        return render_template('no_active_event.html', photographer=photographer)


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

        selfie_video    = request.files.get('selfie_video')
        selfie_fallback = request.files.get('selfie_fallback')

        is_video    = bool(selfie_video and selfie_video.filename)
        selfie_file = selfie_video if is_video else (
            selfie_fallback if (selfie_fallback and selfie_fallback.filename) else None
        )

        # Fallback path still needs the JPG/PNG check; the video path is
        # programmatically generated by MediaRecorder, not user-picked, so
        # it doesn't go through the same extension allowlist.
        if not selfie_file or (not is_video and not allowed_file(selfie_file.filename)):
            flash("Please complete face verification or upload a valid JPG/PNG photo.", "danger")
            return redirect(request.url)

        ext = 'webm' if is_video else 'jpg'
        content_type = selfie_file.mimetype or ('video/webm' if is_video else 'image/jpeg')
        filename = secure_filename(f"{uuid4()}.{ext}")
        storage_path = f"guests/{event_name}/{filename}"
        saved_path = storage.upload_fileobj(storage_path, selfie_file, content_type=content_type)

        gallery_token = str(uuid4())
        guest = Guest(
            name               = name,
            email              = email,
            event_name         = event_name,
            selfie_center_path = saved_path,
            selfie_is_video    = is_video,
            gallery_token      = gallery_token
        )
        db.session.add(guest)
        db.session.commit()

        return render_template('guest_success.html', guest_name=name)

    return render_template('guest_register.html', event_name=event_name,
                           venue=event.venue, event_date=event.date,
                           event_time=event.event_time)


# ─────────────────────────────────────────────
# Routes – Photo Upload
# ─────────────────────────────────────────────
@app.route('/upload/<event_name>', methods=['GET', 'POST'])
def upload_photos(event_name):
    if 'email' not in session:
        return redirect(url_for('photographer_login'))

    if request.method == 'POST':
        files = request.files.getlist('photos')

        saved_count   = 0
        skipped_count = 0
        uploaded_filenames = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                storage.upload_fileobj(f"photos/{event_name}/{filename}", file)
                uploaded_filenames.append(filename)
                saved_count += 1
            else:
                skipped_count += 1

        if uploaded_filenames:
            # Only insert filenames we haven't seen before for this event —
            # re-uploading an existing filename just refreshes the Storage
            # object without resetting its already-matched status.
            existing = {
                row.filename for row in EventPhoto.query
                    .filter_by(event_name=event_name)
                    .filter(EventPhoto.filename.in_(uploaded_filenames))
                    .all()
            }
            new_filenames = [f for f in uploaded_filenames if f not in existing]
            for filename in new_filenames:
                db.session.add(EventPhoto(event_name=event_name, filename=filename, matched=False))

            if new_filenames:
                event = Event.query.filter_by(name=event_name).first()
                if event:
                    event.photo_count = (event.photo_count or 0) + len(new_filenames)

            db.session.commit()

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

    filenames = [f for f in storage.list_files(f"photos/{event_name}") if allowed_file(f)]
    if not filenames:
        return f"No photos found for event: {event_name}", 404

    photos = [
        {"name": f, "url": storage.get_public_url(f"photos/{event_name}/{f}")}
        for f in filenames
    ]

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

    if not storage.list_files(f"photos/{event_name}"):
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
    guest = Guest.query.filter_by(gallery_token=uuid).first_or_404()
    filenames = [f for f in storage.list_files(f"matches/{uuid}") if allowed_file(f)]

    if not filenames:
        return render_template('gallery.html', guest=guest, photos=[], message="No matched photos yet. Please check back soon.")

    photos = [storage.get_public_url(f"matches/{uuid}/{f}") for f in filenames]
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