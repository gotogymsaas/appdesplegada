import argparse
import mimetypes
import os

from azure.storage.blob import BlobServiceClient, ContentSettings


def guess_content_type(name):
    content_type, _ = mimetypes.guess_type(name)
    return content_type or "application/octet-stream"


def main():
    parser = argparse.ArgumentParser(
        description="Fix blob Content-Type metadata for media files."
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not update blobs")
    args = parser.parse_args()

    account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "").strip()
    key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "").strip()
    container = os.getenv("AZURE_STORAGE_CONTAINER", "media").strip()

    if not account or not key:
        raise SystemExit("Missing AZURE_STORAGE_ACCOUNT_NAME or AZURE_STORAGE_ACCOUNT_KEY")

    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=key,
    )
    container_client = service.get_container_client(container)

    for blob in container_client.list_blobs():
        blob_client = container_client.get_blob_client(blob.name)
        props = blob_client.get_blob_properties()
        current = props.content_settings.content_type
        guessed = guess_content_type(blob.name)

        if current and current not in ("text/plain", "application/octet-stream"):
            continue

        if args.dry_run:
            print(f"[DRY RUN] {blob.name}: {current} -> {guessed}")
            continue

        content_settings = ContentSettings(
            content_type=guessed,
            cache_control="public, max-age=31536000",
        )
        blob_client.set_http_headers(content_settings=content_settings)
        print(f"Updated {blob.name}: {current} -> {guessed}")


if __name__ == "__main__":
    main()
