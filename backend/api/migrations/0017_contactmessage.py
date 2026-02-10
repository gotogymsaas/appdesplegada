from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0016_userdocument"),
    ]

    operations = [
        migrations.CreateModel(
            name="ContactMessage",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=160)),
                ("email", models.EmailField(max_length=254)),
                ("subject", models.CharField(max_length=200)),
                ("message", models.TextField()),
                ("status", models.CharField(choices=[("received", "Recibido"), ("responded", "Respondido")], default="received", max_length=20)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("responded_at", models.DateTimeField(blank=True, null=True)),
            ],
        ),
    ]
