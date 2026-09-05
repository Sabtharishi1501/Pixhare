from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Photographer(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)          # NEW
    studio_name   = db.Column(db.String(120), nullable=False)          # NEW
    email         = db.Column(db.String(120), unique=True, nullable=False)
    otp           = db.Column(db.String(6))
    password      = db.Column(db.String(200), nullable=False)
    is_verified   = db.Column(db.Boolean, default=False)
    registered_on = db.Column(db.DateTime, default=datetime.utcnow)
    events        = db.relationship('Event', backref='photographer', lazy=True)

    # One QR per photographer (not per event) — guests scan this once and
    # get routed to whichever event(s) are live today. scan_token is the
    # unguessable identifier encoded in the QR URL, in the same style as
    # Guest.gallery_token.
    scan_token    = db.Column(db.String(64), unique=True)
    qr_url        = db.Column(db.String(300))

class Guest(db.Model):
    id                  = db.Column(db.Integer, primary_key=True)
    name                = db.Column(db.String(120))
    email               = db.Column(db.String(120))
    event_name          = db.Column(db.String(120))
    selfie_center_path  = db.Column(db.String(300))   # video OR fallback image path
    selfie_left_path    = db.Column(db.String(300))   # unused going forward, kept for old rows
    selfie_right_path   = db.Column(db.String(300))   # unused going forward, kept for old rows
    selfie_is_video     = db.Column(db.Boolean, default=False)
    gallery_token       = db.Column(db.String(120), unique=True)
    gallery_sent_at     = db.Column(db.DateTime)   # set on first email; distinguishes
                                                    # "your gallery is ready" from
                                                    # "new photos added" on later runs

class EventPhoto(db.Model):
    """Tracks each uploaded event photo's matching status, so a repeat
    'Send Gallery' run only processes photos uploaded since the last run
    instead of re-embedding the whole event every time."""
    id          = db.Column(db.Integer, primary_key=True)
    event_name  = db.Column(db.String(120), nullable=False)
    filename    = db.Column(db.String(300), nullable=False)
    matched     = db.Column(db.Boolean, default=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('event_name', 'filename', name='uq_event_photo'),
    )

class PhotoFaceEmbedding(db.Model):
    """One row per detected face per photo (a group photo produces several).
    Cached so a photo's face embeddings only ever get computed once — a
    late-registering guest can be matched against the event's full photo
    history by reading these back, without re-running RetinaFace/ArcFace
    on photos that were already processed in an earlier 'Send Gallery' run."""
    id         = db.Column(db.Integer, primary_key=True)
    event_name = db.Column(db.String(120), nullable=False)
    filename   = db.Column(db.String(300), nullable=False)
    embedding  = db.Column(db.Text, nullable=False)
    confidence = db.Column(db.Float)

class Event(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    name            = db.Column(db.String(100), nullable=False, unique=True)
    date            = db.Column(db.String(100), nullable=False)
    venue           = db.Column(db.String(200))
    event_time      = db.Column(db.String(20))
    photographer_id = db.Column(db.Integer, db.ForeignKey('photographer.id'))
    photo_count     = db.Column(db.Integer, default=0)