from django.db import migrations


class Migration(migrations.Migration):
    """No-op migration kept for historical compatibility.

    En producción existe una rama de migraciones con este nombre.
    Si se elimina del repo, Django puede detectar hojas múltiples y abortar.

    Esta migración no aplica cambios de esquema: solo preserva el grafo.
    """

    dependencies = [
        ("api", "0022_auditlog"),
    ]

    operations = []
