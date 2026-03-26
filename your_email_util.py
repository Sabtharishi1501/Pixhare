import smtplib
from email.message import EmailMessage

def send_otp_email(to_email, otp):
    msg = EmailMessage()
    msg['Subject'] = 'Your OTP for PhotoShare AI'
    msg['From'] = 'your@email.com'
    msg['To'] = to_email
    msg.set_content(f'Your OTP for registration is: {otp}')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('your@email.com', 'your_password_or_app_password')
        smtp.send_message(msg)
