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

class Guest(db.Model):
    id                  = db.Column(db.Integer, primary_key=True)
    name                = db.Column(db.String(120))
    email               = db.Column(db.String(120))
    event_name          = db.Column(db.String(120))
    selfie_center_path  = db.Column(db.String(300))
    selfie_left_path    = db.Column(db.String(300))
    selfie_right_path   = db.Column(db.String(300))
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
    embedding  = db.Column(db.Text, nullable=False)   # JSON list of 512 floats
    confidence = db.Column(db.Float)

class Event(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    name            = db.Column(db.String(100), nullable=False)
    date            = db.Column(db.String(100), nullable=False)
    qr_filename     = db.Column(db.String(200))
    photographer_id = db.Column(db.Integer, db.ForeignKey('photographer.id'))