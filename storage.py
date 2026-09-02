"""
Supabase Storage helper.

Vercel's container Functions have no shared, persistent filesystem — a file
saved to local disk by one instance isn't visible to a different instance
that later handles a gallery view or a matching job. This module replaces
every local `static/...` read/write with calls to a single Supabase Storage
bucket, addressed by path instead of by local folder.

SETUP REQUIRED (one-time, in the Supabase dashboard):
  Storage → New bucket → name it exactly "pixhare" → toggle "Public bucket" ON.
  Public, because gallery links already work by an unguessable UUID token
  (the same trust model the app already uses) rather than per-file auth.

Path layout inside the bucket:
  qrcodes/{event_name}_qr.png
  guests/{event_name}/{uuid}.jpg
  photos/{event_name}/{filename}
  matches/{gallery_token}/{filename}
"""
from supabase import create_client
from config import Config

BUCKET = "pixhare"

_client = None


def get_client():
    global _client
    if _client is None:
        _client = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_ROLE_KEY)
    return _client


def upload_fileobj(path, file_storage, content_type="image/jpeg"):
    """Upload a Werkzeug FileStorage (from request.files) to Storage.
    Returns the storage path on success, None if there was nothing to upload."""
    if not file_storage:
        return None
    data = file_storage.read()
    return upload_bytes(path, data, content_type)


def upload_bytes(path, data, content_type="image/jpeg"):
    client = get_client()
    client.storage.from_(BUCKET).upload(
        path, data,
        file_options={"content-type": content_type, "upsert": "true"}
    )
    return path


def download_bytes(path):
    """Return the raw bytes for a stored file, or None if missing/unreadable."""
    if not path:
        return None
    try:
        return get_client().storage.from_(BUCKET).download(path)
    except Exception as e:
        print(f"[storage] ⚠️ Could not download {path}: {e}")
        return None


def get_public_url(path):
    if not path:
        return None
    return get_client().storage.from_(BUCKET).get_public_url(path)


def list_files(prefix):
    """Return bare filenames (not full paths) directly under a prefix."""
    try:
        entries = get_client().storage.from_(BUCKET).list(prefix)
        return [e["name"] for e in entries if e.get("id") is not None]
    except Exception as e:
        print(f"[storage] ⚠️ Could not list {prefix}: {e}")
        return []


def delete_file(path):
    if path:
        get_client().storage.from_(BUCKET).remove([path])


def delete_prefix(prefix):
    """Delete every file directly under a prefix (Storage has no recursive
    delete, so list then remove as a batch)."""
    names = list_files(prefix)
    if not names:
        return
    paths = [f"{prefix.rstrip('/')}/{name}" for name in names]
    get_client().storage.from_(BUCKET).remove(paths)


def copy_file(src_path, dest_path):
    get_client().storage.from_(BUCKET).copy(src_path, dest_path)


def file_exists(path):
    prefix = "/".join(path.split("/")[:-1])
    name = path.split("/")[-1]
    return name in list_files(prefix)