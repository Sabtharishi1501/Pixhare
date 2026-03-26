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
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120))
    email         = db.Column(db.String(120))
    event_name    = db.Column(db.String(120))
    selfie_path   = db.Column(db.String(300))
    gallery_token = db.Column(db.String(120), unique=True)

class Event(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    name            = db.Column(db.String(100), nullable=False)
    date            = db.Column(db.String(100), nullable=False)
    qr_filename     = db.Column(db.String(200))
    photographer_id = db.Column(db.Integer, db.ForeignKey('photographer.id'))