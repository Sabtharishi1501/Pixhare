# ─────────────────────────────────────────────
# knowledge_base.py
# Pixhare FAQ & guide documents for RAG
# ─────────────────────────────────────────────

DOCUMENTS = [
    {
        "id": "what_is_pixhare",
        "text": """
Pixhare is an AI-powered event photo sharing platform.
It helps photographers manage events, upload bulk photos, and automatically
deliver personalised photo galleries to guests using face recognition AI.
Guests register by scanning a QR code and uploading a selfie.
DeepFace AI matches every guest's face to the event photos
and sends each guest a private gallery link via email instantly.
        """
    },
    {
        "id": "how_it_works",
        "text": """
How Pixhare works step by step:
1. Photographer creates an event on the dashboard and receives a unique QR code poster.
2. Photographer shares or displays the QR code at the event venue.
3. Guests scan the QR code with their phone, enter name and email, and upload a selfie.
4. After the event, photographer uploads all event photos to Pixhare.
5. Photographer clicks Match and Send Galleries button.
6. Pixhare AI using DeepFace ArcFace model scans every photo and matches faces to guests.
7. Each guest receives a private email with a link to their personalised gallery.
        """
    },
    {
        "id": "photographer_registration",
        "text": """
How to register as a photographer on Pixhare:
1. Go to the Register page from the navbar.
2. Enter your full name, studio name, email address, and a strong password.
3. Click Send OTP. A 6-digit OTP will be sent to your email within seconds.
4. Enter the OTP on the verification page. OTP is valid for 5 minutes.
5. Once verified your account is created and you can login.
If you did not receive the OTP check your spam folder.
If the OTP expired click Go Back and register again.
        """
    },
    {
        "id": "photographer_login",
        "text": """
How to login to Pixhare as a photographer:
1. Go to the Login page from the navbar.
2. Enter your registered email and password.
3. Click Login.
If you forgot your password click Forgot Password link on the login page,
enter your email, verify with an OTP sent to your email, and set a new password.
After successful login you will be redirected to your photographer dashboard.
        """
    },
    {
        "id": "forgot_password",
        "text": """
How to reset your forgotten password on Pixhare:
1. Click Forgot Password on the login page.
2. Enter your registered email address and click Send OTP.
3. Check your email for a 6-digit OTP.
4. Enter the OTP on the verification page. OTP is valid for 5 minutes.
5. After OTP is verified you will be taken to the Reset Password page.
6. Enter your new password and confirm it.
7. Click Reset Password. You can now login with your new password.
        """
    },
    {
        "id": "create_event",
        "text": """
How to create an event on Pixhare:
1. Login to your photographer dashboard.
2. On the left sidebar enter the event name and event date.
3. Click Create Event.
4. A unique QR code poster is automatically generated for the event.
5. Click the event card to expand it and click QR Code button to view and download the poster.
6. Display the QR poster at the venue for guests to scan.
Each event has its own separate QR code so guests always register for the correct event.
You can have multiple events at the same time.
        """
    },
    {
        "id": "upload_photos",
        "text": """
How to upload photos to an event on Pixhare:
1. Go to your dashboard and click on the event card to expand it.
2. Click Upload Photos button.
3. Drag and drop your photos or click to browse and select files.
4. Only JPG, JPEG, and PNG files are accepted.
5. Maximum file size is 16MB per photo.
6. You can upload multiple photos at once in bulk.
7. You can upload more photos later by clicking Upload More.
After uploading you can view all photos in the View Photos page.
        """
    },
    {
        "id": "delete_photos",
        "text": """
How to delete photos from an event on Pixhare:
1. Go to View Photos page for the event.
2. Click the Delete Photos button in the top bar.
3. The page enters delete mode. Click on photos you want to delete to select them.
4. Selected photos show a red border and a checkmark.
5. Click Delete Selected button when done selecting.
6. A confirmation popup will ask you to confirm.
7. Click Yes Delete to permanently delete the selected photos.
To exit delete mode without deleting click Cancel or Exit Delete Mode.
        """
    },
    {
        "id": "face_matching",
        "text": """
How Pixhare AI face matching works:
Pixhare uses DeepFace with ArcFace model and RetinaFace detector.
These are state of the art face recognition models with very high accuracy.
The matching process:
1. Each guest selfie is converted into a face embedding (mathematical face representation).
2. Every event photo is also converted into face embeddings once and cached.
3. System compares each guest embedding against all photo embeddings using cosine distance.
4. If distance is below threshold 0.40 the photo is a match for that guest.
5. Matched photos are copied to the guest private gallery folder.
Matching runs in the background so the page does not freeze.
Multiple guests are processed in parallel for speed.
Results may take a few minutes for large events.
        """
    },
    {
        "id": "send_gallery",
        "text": """
How to send galleries to guests on Pixhare:
1. After uploading photos click Match and Send Galleries in View Photos page,
   or click Send Gallery on the event card in the dashboard.
2. Pixhare starts AI face matching in the background.
3. Each matched guest receives an email with their private gallery link.
4. The gallery link is unique per guest and shows only their matched photos.
5. Gallery links do not expire.
If a guest does not receive the email check their spam folder.
Make sure the guest registered with a valid email address.
        """
    },
    {
        "id": "guest_registration",
        "text": """
How guests register on Pixhare:
1. Photographer displays a QR code poster at the event venue.
2. Guests scan the QR code with their phone camera. No app download needed.
3. The registration page opens in the browser automatically.
4. Guests enter their full name and email address.
5. Guests upload a selfie by taking a photo with camera or uploading from gallery.
6. After submitting guests see a confirmation screen.
7. Once photographer runs face matching guests receive gallery link via email.
Make sure face is clearly visible in the selfie for best matching accuracy.
The selfie should be a clear frontal face photo with good lighting.
        """
    },
    {
        "id": "gallery",
        "text": """
About guest photo galleries on Pixhare:
Each guest gets a unique private gallery link sent to their email.
The gallery shows only the photos where that specific guest appears.
Gallery links are permanent and do not expire.
Guests can view and download their photos from the gallery.
The gallery is accessible only via the unique private link.
If a guest cannot find their gallery link they should check their spam or junk folder.
If they still cannot find it they should contact the photographer to resend.
        """
    },
    {
        "id": "qr_code",
        "text": """
About Pixhare QR codes and posters:
Each event gets its own unique QR code when the event is created.
The QR code links to the guest registration page for that specific event.
Scanning the QR code with any phone camera opens the registration page instantly.
The QR code is available as a branded Pixhare poster with the event name.
The poster has a dark professional design with the Pixhare logo and instructions.
Photographers can download the poster and print it or display it digitally at the venue.
To download click the QR Code button on the event card then click Download Poster.
        """
    },
    {
        "id": "privacy_security",
        "text": """
Privacy and security on Pixhare:
Guest selfies are only used for face matching and stored securely on the server.
Each guest gallery is private and accessible only via a unique token-based URL.
Gallery links are not shared publicly or with other guests.
Pixhare does not sell or share your data with third parties.
Guest emails are only used to send the gallery link.
Photographers can delete an event and all its data including photos selfies and galleries at any time.
        """
    },
    {
        "id": "dashboard",
        "text": """
About the Pixhare photographer dashboard:
The dashboard shows all your events in a card grid layout.
Click any event card to expand it and see management options.
Each event card shows the event name and date.
Expanded event panel shows:
- QR Code button to view and download the branded poster
- Upload Photos button to add photos to the event
- View Photos button to see all uploaded photos in a grid
- Send Gallery button to start AI matching and email guests
- Delete button to permanently delete the event and all its data
At the bottom of the expanded panel you can see total guests registered and total photos uploaded.
        """
    },
    {
        "id": "troubleshooting",
        "text": """
Common issues and solutions on Pixhare:
Problem: OTP not received.
Solution: Check spam folder. OTP is valid for 5 minutes. If expired go back and register again.

Problem: Guest did not receive gallery email.
Solution: Check spam folder. Make sure guest registered with correct email. Photographer can click Send Gallery again.

Problem: Photos not matching correctly.
Solution: Make sure guest selfie has a clear frontal face with good lighting. Avoid sunglasses or masks in selfie.

Problem: QR code not working.
Solution: Make sure you are using the correct QR code for the specific event. Each event has a different QR.

Problem: Cannot login.
Solution: Check email and password. Use Forgot Password to reset if needed.

Problem: Event not found when scanning QR.
Solution: The event may have been deleted by the photographer. Contact the photographer.

Problem: Upload failing.
Solution: Only JPG PNG files are accepted. Maximum 16MB per file. Try uploading fewer photos at once.
        """
    },
    {
        "id": "studio_profile",
        "text": """
About photographer profile on Pixhare:
When registering photographers provide their full name and studio name.
The studio name appears in the branding.
The photographer name appears in the navbar greeting as Hi Name.
Photographers can have multiple events running simultaneously.
Each photographer only sees their own events on the dashboard.
Events from other photographers are not visible.
        """
    },
]