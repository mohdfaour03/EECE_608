"""Upload a consistent experiment checkpoint to Google Drive from Colab.

Run ``google.colab.auth.authenticate_user()`` once in the notebook before using
this helper. SQLite inputs are copied with SQLite's backup API so the uploaded
file is transactionally consistent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import tempfile


DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"


def _drive_service():
    import google.auth
    from googleapiclient.discovery import build

    credentials, _ = google.auth.default(scopes=[DRIVE_SCOPE])
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _find_file_id(service, name: str) -> str | None:
    escaped_name = name.replace("'", "\\'")
    response = service.files().list(
        q=f"name = '{escaped_name}' and trashed = false",
        spaces="drive",
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=10,
    ).execute()
    files = response.get("files", [])
    return files[0]["id"] if files else None


def _snapshot_source(source: Path, temporary_dir: Path) -> Path:
    if source.suffix not in {".db", ".sqlite", ".sqlite3"}:
        return source

    snapshot = temporary_dir / source.name
    with sqlite3.connect(source) as source_db, sqlite3.connect(snapshot) as snapshot_db:
        source_db.backup(snapshot_db)
    return snapshot


def upload(source: Path, drive_name: str) -> str:
    from googleapiclient.http import MediaFileUpload

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    service = _drive_service()
    with tempfile.TemporaryDirectory(prefix="colab-checkpoint-") as temporary:
        upload_path = _snapshot_source(source, Path(temporary))
        media = MediaFileUpload(str(upload_path), resumable=True)
        file_id = _find_file_id(service, drive_name)
        if file_id:
            result = service.files().update(
                fileId=file_id,
                media_body=media,
                fields="id, name, size, modifiedTime",
            ).execute()
        else:
            result = service.files().create(
                body={"name": drive_name},
                media_body=media,
                fields="id, name, size, modifiedTime",
            ).execute()
    print(
        "checkpoint_uploaded "
        f"id={result['id']} name={result['name']} size={result.get('size', '?')} "
        f"modified={result.get('modifiedTime', '?')}"
    )
    return result["id"]


def download(drive_name: str, destination: Path) -> str:
    from googleapiclient.http import MediaIoBaseDownload

    service = _drive_service()
    file_id = _find_file_id(service, drive_name)
    if not file_id:
        print(f"checkpoint_not_found name={drive_name}")
        return ""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with destination.open("wb") as output:
        downloader = MediaIoBaseDownload(output, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    print(f"checkpoint_downloaded id={file_id} name={drive_name} path={destination}")
    return file_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--source", type=Path, required=True)
    upload_parser.add_argument("--name", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--name", required=True)
    download_parser.add_argument("--destination", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "upload":
        upload(args.source, args.name)
    else:
        download(args.name, args.destination)


if __name__ == "__main__":
    main()
