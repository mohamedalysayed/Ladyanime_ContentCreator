# rm -f data/credentials/token.json
# rm -rf ~/.cache/google-auth
# python3 tools/upload_shorts.py --auth-only

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# --- CONFIG ---
#SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
# SCOPES = [
#     "https://www.googleapis.com/auth/youtube.upload",
#     "https://www.googleapis.com/auth/youtube.readonly",
# ]
SCOPES = ["https://www.googleapis.com/auth/youtube"]

ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "data" / "output" / "shorts"
STATE_FILE = SHORTS_DIR / "uploaded.json"
# CLIENT_SECRETS = ROOT / "client_secrets.json"   # adjust if needed
# TOKEN_FILE = ROOT / "youtube_token.json"        # stored after first login

CREDS_DIR = ROOT / "data" / "credentials"
CLIENT_SECRETS = CREDS_DIR / "client_secret.json"
TOKEN_FILE = CREDS_DIR / "token.json"

# 2 uploads/day at these local times:
SLOT_TIMES = ["12:00", "18:00"]  # HH:MM

@dataclass
class UploadItem:
    path: Path
    title: str
    description: str
    tags: list[str]
    publish_at: datetime  # local time


def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"uploaded_files": {}, "last_run": None}


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

CREDS_DIR.mkdir(parents=True, exist_ok=True)
def _get_service():
    # First run will open browser for OAuth, then cache token.
    # Using InstalledAppFlow keeps it simple for local automation.
    if not CLIENT_SECRETS.exists():
        raise FileNotFoundError(
            f"Missing OAuth client secret file:\n{CLIENT_SECRETS}\n"
            "Download it from Google Cloud Console → OAuth credentials."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
    creds = None

    if TOKEN_FILE.exists():
        # load cached credentials (google-auth will refresh if possible)
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=creds)

EXPECTED_HANDLE = "@ladyanimee"   # or channel title if you prefer

def _log_channel_info(youtube):
    try:
        resp = youtube.channels().list(
            part="snippet",
            mine=True
        ).execute()

        if not resp.get("items"):
            print("⚠️ Brand channel detected (mine=true returned empty). Upload will still work.")
            return

        ch = resp["items"][0]
        print(f"🔐 OAuth channel: {ch['snippet']['title']}")

    except Exception as e:
        print("⚠️ Channel verification skipped:", e)

def _verify_channel(youtube):
    resp = youtube.channels().list(
        part="snippet",
        mine=True
    ).execute()

    if not resp.get("items"):
        raise RuntimeError(
            "❌ OAuth has NO YouTube channel attached."
        )

    ch = resp["items"][0]
    title = ch["snippet"]["title"]
    custom_url = ch["snippet"].get("customUrl", "")

    print(f"🔐 OAuth channel detected: {title}")

    if EXPECTED_HANDLE.lower().lstrip("@") not in custom_url.lower():
        raise RuntimeError(
            f"❌ WRONG CHANNEL SELECTED.\n"
            f"Expected: {EXPECTED_HANDLE}\n"
            f"Got: {title} ({custom_url})\n\n"
            f"➡ Re-run --auth-only and select the LadyAnime channel."
        )

    print("✅ OAuth locked to LadyAnime channel.")

def _next_publish_times(today: datetime) -> list[datetime]:
    # schedule for "today" if still in the future, else schedule starting tomorrow
    slots = []
    for hhmm in SLOT_TIMES:
        h, m = map(int, hhmm.split(":"))
        dt = today.replace(hour=h, minute=m, second=0, microsecond=0)
        slots.append(dt)

    now = datetime.now()
    if all(t <= now for t in slots):
        base = today + timedelta(days=1)
        slots = [base.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0) for t in slots]
    else:
        # keep only future slots today; if only one left, second goes tomorrow first slot
        future = [t for t in slots if t > now]
        if len(future) == 2:
            return future
        if len(future) == 1:
            tomorrow = today + timedelta(days=1)
            return [future[0], tomorrow.replace(hour=int(SLOT_TIMES[0].split(":")[0]),
                                                minute=int(SLOT_TIMES[0].split(":")[1]),
                                                second=0, microsecond=0)]
    return slots


def _pick_two_not_uploaded(state: dict[str, Any]) -> list[Path]:
    uploaded = set(state["uploaded_files"].keys())
    candidates = sorted([p for p in SHORTS_DIR.glob("*.mp4") if p.name not in uploaded])
    return candidates[:2]


def _build_items(files: list[Path], publish_times: list[datetime]) -> list[UploadItem]:
    items: list[UploadItem] = []
    for p, t in zip(files, publish_times):
        # Customize titles/desc however you like
        title = p.stem.replace("_", " ")[:95]  # keep it short-ish
        description = "#Shorts\nLadyAnime\n"
        tags = ["Shorts", "LadyAnime", "Anime"]
        items.append(UploadItem(path=p, title=title, description=description, tags=tags, publish_at=t))
    return items


def _to_rfc3339(dt: datetime) -> str:
    # RFC3339 with local offset (YouTube expects RFC3339 for publishAt)
    # If your system timezone is set correctly, this is fine.
    # Example: 2025-01-01T18:00:00+01:00
    offset = dt.astimezone().strftime("%z")
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f"{offset[:3]}:{offset[3:]}"


def upload_one(youtube, item: UploadItem, privacy_status: str = "private") -> str:
    body = {
        "snippet": {
            "title": item.title,
            "description": item.description,
            "tags": item.tags,
            "categoryId": "24",  # Entertainment (adjust if you want)
        },
        "status": {
            "privacyStatus": privacy_status,
            "publishAt": _to_rfc3339(item.publish_at),
        },
    }

    media = MediaFileUpload(str(item.path), mimetype="video/*", resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"Uploading {item.path.name}: {pct}%")

    # response includes uploaded video resource
    return response["id"]


#def main():
def main(dry_run: bool = False):
    state = _load_state()
    files = _pick_two_not_uploaded(state)

    if not files:
        print("No new shorts to upload.")
        return

    today = datetime.now()
    publish_times = _next_publish_times(today)
    items = _build_items(files, publish_times)

    if dry_run:
        print("🧪 DRY RUN — no uploads performed\n")
        for item in items:
            print(f"{item.path.name}")
            print(f"  → scheduled at {item.publish_at}")
            print(f"  → title: {item.title}")
            print()
        return

    youtube = _get_service()
    _log_channel_info(youtube)

    # NOTE:
    # publishAt is supported on videos.insert status. :contentReference[oaicite:3]{index=3}
    # Quota cost is 100 units per call. :contentReference[oaicite:4]{index=4}

    for item in items:
        vid = upload_one(youtube, item, privacy_status="private")  # switch to "public" if your project allows scheduling
        state["uploaded_files"][item.path.name] = {
            "video_id": vid,
            "publish_at": _to_rfc3339(item.publish_at),
            "uploaded_at": datetime.now().isoformat(),
        }
        print(f"Uploaded: {item.path.name} → video id {vid}")

    state["last_run"] = datetime.now().isoformat()
    _save_state(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LadyAnime YouTube Shorts Uploader")
    parser.add_argument("--auth-only", action="store_true", help="Only authenticate with YouTube")
    parser.add_argument("--dry-run", action="store_true", help="Preview uploads without uploading")

    args = parser.parse_args()

    if args.auth_only:
        _get_service()
        print("✅ Authentication successful. Token stored.")
    else:
        main(dry_run=args.dry_run)

