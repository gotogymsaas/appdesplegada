import mimetypes

from django.conf import settings
from storages.backends.azure_storage import AzureStorage


class MediaAzureStorage(AzureStorage):
    account_name = settings.AZURE_STORAGE_ACCOUNT_NAME
    account_key = settings.AZURE_STORAGE_ACCOUNT_KEY
    azure_container = settings.AZURE_STORAGE_CONTAINER
    custom_domain = settings.AZURE_STORAGE_CUSTOM_DOMAIN or None
    overwrite_files = False
    expiration_secs = getattr(settings, "AZURE_SAS_EXPIRATION", 3600)
    cache_control = "public, max-age=31536000"
    default_content_type = "application/octet-stream"

    def _save(self, name, content):
        if not getattr(content, "content_type", None):
            guessed, _ = mimetypes.guess_type(name)
            if guessed:
                content.content_type = guessed
        return super()._save(name, content)
