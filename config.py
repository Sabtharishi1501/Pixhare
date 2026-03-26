import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///photo_share.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'your_secret_key'
    
    EMAIL_USER = 'photshareai@gmail.com'
    EMAIL_PASS = 'ezecqznqxlvrbupr'
    
    HF_TOKEN = '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'
    GROQ_API_KEY = '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'