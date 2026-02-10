from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0017_contactmessage"),
    ]

    operations = [
        migrations.CreateModel(
            name="PushToken",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("token", models.CharField(max_length=512, unique=True)),
                ("platform", models.CharField(choices=[("android", "Android"), ("ios", "iOS"), ("web", "Web"), ("unknown", "Unknown")], default="unknown", max_length=20)),
                ("device_id", models.CharField(blank=True, default="", max_length=120)),
                ("active", models.BooleanField(default=True)),
                ("last_seen_at", models.DateTimeField(auto_now=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("user", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="push_tokens", to="api.user")),
            ],
        ),
    ]
