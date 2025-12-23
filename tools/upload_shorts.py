# rm -f data/credentials/token.json
# rm -rf ~/.cache/google-auth
# python3 tools/upload_shorts.py --auth-only
# python3 tools/upload_shorts.py --rebuild-state
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from zoneinfo import ZoneInfo

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# --- CONFIG ---
SCOPES = ["https://www.googleapis.com/auth/youtube"]

ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "data" / "output" / "shorts"
STATE_FILE = SHORTS_DIR / "uploaded.json"

CREDS_DIR = ROOT / "data" / "credentials"
CLIENT_SECRETS = CREDS_DIR / "client_secret.json"
TOKEN_FILE = CREDS_DIR / "token.json"

TZ = ZoneInfo("Europe/Zurich")  # CET/CEST safe
SLOT_TIMES = ["00:00", "12:00"]  # 2 uploads/day as requested

EXPECTED_HANDLE = "@ladyanimee"  # keep your safeguard

GENERIC_DESCRIPTION = "#ladyAnime #anime #Shorts\n"
GENERIC_TAGS = ["LadyAnime", "anime", "Shorts"]


@dataclass
class UploadItem:
    path: Path
    title: str
    description: str
    tags: list[str]
    publish_at: datetime  # tz-aware


# -------------------------
# State (atomic + safe)
# -------------------------
def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "uploaded_files": {},   # filename -> metadata
        "pending_meta": {},     # filename -> {"title": "..."} optional
        "base_title": "",       # global default title
        "last_run": None,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _save_state(state: dict[str, Any]) -> None:
    # rotate backups
    try:
        if STATE_FILE.exists():
            ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
            backup_dir = STATE_FILE.parent / "state_backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"uploaded_{ts}.json"
            backup_path.write_text(STATE_FILE.read_text(encoding="utf-8"), encoding="utf-8")

            # keep only last 20 backups
            backups = sorted(backup_dir.glob("uploaded_*.json"))
            for old in backups[:-20]:
                old.unlink(missing_ok=True)
    except Exception:
        pass

    _atomic_write_json(STATE_FILE, state)

def rebuild_state_from_youtube(youtube, max_results: int = 200) -> dict[str, Any]:
    """
    Rebuild uploaded_files from YouTube video list.
    This won’t perfectly map to filenames (YouTube doesn't store local filename),
    but it restores video_id + publishAt/title so your channel history is preserved.
    """
    state = _load_state()
    state.setdefault("uploaded_files", {})

    # Get uploads playlist
    ch = youtube.channels().list(part="contentDetails", mine=True).execute()
    if not ch.get("items"):
        return state
    uploads_pl = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Page through playlist items
    next_token = None
    fetched = 0
    while True:
        resp = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_pl,
            maxResults=min(50, max_results - fetched),
            pageToken=next_token
        ).execute()

        for it in resp.get("items", []):
            vid = it["contentDetails"]["videoId"]
            title = it["snippet"]["title"]
            published_at = it["snippet"].get("publishedAt", "")
            state["uploaded_files"][f"YOUTUBE:{vid}"] = {
                "video_id": vid,
                "title": title,
                "published_at": published_at,
                "source": "youtube_rebuild",
            }

        fetched += len(resp.get("items", []))
        next_token = resp.get("nextPageToken")
        if not next_token or fetched >= max_results:
            break

    state["last_run"] = datetime.now(TZ).isoformat(timespec="seconds")
    _save_state(state)
    return state

# -------------------------
# Auth + channel safety
# -------------------------
CREDS_DIR.mkdir(parents=True, exist_ok=True)

def _get_service():
    if not CLIENT_SECRETS.exists():
        raise FileNotFoundError(
            f"Missing OAuth client secret file:\n{CLIENT_SECRETS}\n"
            "Download it from Google Cloud Console → OAuth credentials."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
    creds = None

    if TOKEN_FILE.exists():
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=creds)


def _verify_channel_or_warn(youtube) -> str:
    """
    Mega-safe mode: verify we are on the correct channel when possible.
    Some brand-account setups may return empty for mine=True — in that case we warn.
    """
    try:
        resp = youtube.channels().list(part="snippet", mine=True).execute()
        if not resp.get("items"):
            return "⚠️ Brand-channel behavior: channels().list(mine=True) returned empty. Proceeding cautiously."

        ch = resp["items"][0]
        title = ch["snippet"]["title"]
        custom_url = ch["snippet"].get("customUrl", "")

        # strict check if customUrl exists
        if custom_url and EXPECTED_HANDLE.lower().lstrip("@") not in custom_url.lower():
            raise RuntimeError(
                f"❌ WRONG CHANNEL SELECTED.\n"
                f"Expected: {EXPECTED_HANDLE}\n"
                f"Got: {title} ({custom_url})\n"
                f"➡ Re-run --auth-only and select the LadyAnime channel."
            )

        return f"✅ OAuth channel: {title}"

    except Exception as e:
        return f"⚠️ Channel verification skipped: {e}"


# -------------------------
# Scheduling (2/day, sequential)
# -------------------------
def _parse_hhmm(hhmm: str) -> tuple[int, int]:
    h, m = hhmm.split(":")
    return int(h), int(m)


def _next_available_slot(now_local: datetime) -> datetime:
    """
    Returns the next slot (00:00 or 12:00 local).
    """
    now_local = now_local.astimezone(TZ)

    candidates = []
    for hhmm in SLOT_TIMES:
        h, m = _parse_hhmm(hhmm)
        candidates.append(now_local.replace(hour=h, minute=m, second=0, microsecond=0))

    # if we're before a slot today, take first future
    future = [t for t in candidates if t > now_local]
    if future:
        return min(future)

    # else tomorrow first slot
    tomorrow = now_local + timedelta(days=1)
    h0, m0 = _parse_hhmm(SLOT_TIMES[0])
    return tomorrow.replace(hour=h0, minute=m0, second=0, microsecond=0)


def build_publish_schedule(n: int, start_from: Optional[datetime] = None) -> list[datetime]:
    """
    Build n publish times: 2/day at 00:00, 12:00 CET, sequential days.
    """
    if n <= 0:
        return []

    now_local = (start_from or datetime.now(TZ)).astimezone(TZ)
    first = _next_available_slot(now_local)

    schedule = []
    cur = first

    # helper: list today's slots
    def day_slots(day_dt: datetime) -> list[datetime]:
        out = []
        for hhmm in SLOT_TIMES:
            h, m = _parse_hhmm(hhmm)
            out.append(day_dt.replace(hour=h, minute=m, second=0, microsecond=0))
        return out

    # start day
    day = cur.replace(hour=0, minute=0, second=0, microsecond=0)
    slots = [s for s in day_slots(day) if s >= cur]

    while len(schedule) < n:
        if not slots:
            day = (day + timedelta(days=1))
            slots = day_slots(day)

        schedule.append(slots.pop(0))

    return schedule


def fresh_until_date(total_to_schedule: int) -> Optional[datetime]:
    times = build_publish_schedule(total_to_schedule)
    return times[-1] if times else None


# -------------------------
# Upload (resumable + progress + retry)
# -------------------------
def _to_rfc3339(dt: datetime) -> str:
    # dt is tz-aware -> isoformat includes offset like +01:00
    return dt.astimezone(TZ).isoformat(timespec="seconds")


def upload_one(
    youtube,
    item: UploadItem,
    privacy_status: str,
    made_for_kids: bool,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> str:
    body = {
        "snippet": {
            "title": item.title[:95],
            "description": item.description,
            "tags": item.tags,
            "categoryId": "24",  # Entertainment
        },
        "status": {
            "privacyStatus": privacy_status,
            "publishAt": _to_rfc3339(item.publish_at),
            "selfDeclaredMadeForKids": bool(made_for_kids),
        },
    }

    media = MediaFileUpload(str(item.path), mimetype="video/*", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    last_progress = 0.0

    while response is None:
        status, response = request.next_chunk()
        if status:
            frac = float(status.progress())
            # avoid noisy spam
            if frac - last_progress >= 0.01:
                last_progress = frac
                if progress_cb:
                    progress_cb(frac, f"Uploading {item.path.name} ({int(frac*100)}%)")

    return response["id"]


def _retryable_upload(
    youtube,
    item: UploadItem,
    privacy_status: str,
    made_for_kids: bool,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    max_attempts: int = 5,
) -> str:
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return upload_one(
                youtube,
                item=item,
                privacy_status=privacy_status,
                made_for_kids=made_for_kids,
                progress_cb=progress_cb,
            )
        except HttpError as e:
            # retry on 5xx and rate limits
            code = getattr(e.resp, "status", None)
            if code in (500, 502, 503, 504, 429) and attempt < max_attempts:
                if progress_cb:
                    progress_cb(0.0, f"Retry {attempt}/{max_attempts} after HTTP {code}…")
                time.sleep(delay)
                delay *= 2
                continue
            raise

def _filter_selected_against_disk(selected_files: list[str]) -> list[str]:
    """
    Safety: keep only filenames that exist in SHORTS_DIR.
    Prevents uploading anything unexpected.
    """
    existing = {p.name for p in SHORTS_DIR.glob("*.mp4")}
    return [name for name in selected_files if name in existing]

def pick_not_uploaded(state: dict[str, Any], selected: Optional[list[str]] = None) -> list[Path]:
    uploaded = set(state["uploaded_files"].keys())
    all_candidates = sorted([p for p in SHORTS_DIR.glob("*.mp4") if p.name not in uploaded])

    if selected:
        selected_set = set(selected)
        all_candidates = [p for p in all_candidates if p.name in selected_set]

    return all_candidates

def build_items(
    files: list[Path],
    publish_times: list[datetime],
    base_title: str,
    per_file_titles: Optional[dict[str, str]] = None,
    description: str = GENERIC_DESCRIPTION,
    tags: Optional[list[str]] = None,
) -> list[UploadItem]:
    tags = tags or GENERIC_TAGS
    per_file_titles = per_file_titles or {}

    items: list[UploadItem] = []
    for p, t in zip(files, publish_times):
        title = per_file_titles.get(p.name) or base_title or p.stem.replace("_", " ")
        title = title.strip()[:95]
        items.append(
            UploadItem(
                path=p,
                title=title,
                description=description,
                tags=tags,
                publish_at=t,
            )
        )
    return items


def upload_many(
    selected_files: Optional[list[str]] = None,
    schedule_all_remaining: bool = False,
    base_title: str = "",
    per_file_titles: Optional[dict[str, str]] = None,
    description: str = GENERIC_DESCRIPTION,
    tags: Optional[list[str]] = None,
    privacy_status: str = "private",
    made_for_kids: bool = False,
    dry_run: bool = False,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """
    Returns a result dict with logs + updated state info.
    """
    state = _load_state()

    # persist base title
    if base_title:
        state["base_title"] = base_title

    if selected_files:
        selected_files = _filter_selected_against_disk(selected_files)

    # determine files to upload
    if schedule_all_remaining:
        files = pick_not_uploaded(state, selected=None)
    else:
        files = pick_not_uploaded(state, selected=selected_files or [])

    if not files:
        return {"ok": True, "log": "No new shorts to upload.", "scheduled_until": None}

    publish_times = build_publish_schedule(len(files))
    items = build_items(
        files=files,
        publish_times=publish_times,
        base_title=state.get("base_title", ""),
        per_file_titles=per_file_titles,
        description=description,
        tags=tags,
    )

    scheduled_until = publish_times[-1]

    if dry_run:
        lines = ["🧪 DRY RUN — no uploads performed", ""]
        for it in items:
            lines.append(f"{it.path.name}")
            lines.append(f"  → scheduled at {it.publish_at.isoformat(timespec='minutes')}")
            lines.append(f"  → title: {it.title}")
            lines.append(f"  → privacy: {privacy_status}")
            lines.append("")
        return {"ok": True, "log": "\n".join(lines), "scheduled_until": scheduled_until.isoformat()}

    youtube = _get_service()
    ch_msg = _verify_channel_or_warn(youtube)

    lines = [ch_msg, ""]

    for idx, item in enumerate(items, start=1):
        if progress_cb:
            progress_cb(0.0, f"[{idx}/{len(items)}] Starting {item.path.name}…")

        # vid = _retryable_upload(
        #     youtube=youtube,
        #     item=item,
        #     privacy_status=privacy_status,
        #     made_for_kids=made_for_kids,
        #     progress_cb=progress_cb,
        # )

        try:
            vid = _retryable_upload(
                youtube=youtube,
                item=item,
                privacy_status=privacy_status,
                made_for_kids=made_for_kids,
                progress_cb=progress_cb,
            )
        except HttpError as e:
            reason = str(e)
            if "quotaExceeded" in reason:
                lines.append("⛔ YouTube API quota exceeded.")
                lines.append("⏸ Upload paused safely.")
                lines.append("▶ Resume tomorrow — remaining shorts are preserved.")
                _save_state(state)
                return {
                    "ok": False,
                    "log": "\n".join(lines),
                    "scheduled_until": None,
                    "quota_exceeded": True,
                }
            raise

        # state["uploaded_files"][item.path.name] = {
        #     "video_id": vid,
        #     "publish_at": _to_rfc3339(item.publish_at),
        #     "uploaded_at": datetime.now(TZ).isoformat(timespec="seconds"),
        #     "title": item.title,
        # }
        state["uploaded_files"][item.path.name] = {
            "video_id": vid,
            "publish_at": _to_rfc3339(item.publish_at),
            "uploaded_at": datetime.now(TZ).isoformat(timespec="seconds"),
            "title": item.title,
            "privacy": privacy_status,
        }


        lines.append(f"✅ Uploaded: {item.path.name} → {vid} (publishAt {state['uploaded_files'][item.path.name]['publish_at']})")

        # save after each success (super safe)
        state["last_run"] = datetime.now(TZ).isoformat(timespec="seconds")
        _save_state(state)

    return {
        "ok": True,
        "log": "\n".join(lines),
        "scheduled_until": scheduled_until.isoformat(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LadyAnime YouTube Shorts Uploader")
    parser.add_argument("--auth-only", action="store_true", help="Only authenticate with YouTube")
    parser.add_argument("--dry-run", action="store_true", help="Preview uploads without uploading")
    parser.add_argument("--rebuild-state", action="store_true", help="Rebuild uploaded.json from YouTube uploads list")
    args = parser.parse_args()

    if args.rebuild_state:
        yt = _get_service()
        msg = _verify_channel_or_warn(yt)
        print(msg)
        st = rebuild_state_from_youtube(yt)
        print(f"✅ Rebuilt state: {len(st.get('uploaded_files', {}))} entries")

    if args.auth_only:
        _get_service()
        print("✅ Authentication successful. Token stored.")
    else:
        res = upload_many(dry_run=args.dry_run)
        print(res["log"])
