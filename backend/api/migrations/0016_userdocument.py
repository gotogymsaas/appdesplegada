from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0015_user_billing_trial_fields"),
    ]

    operations = [
        migrations.CreateModel(
            name="UserDocument",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("doc_type", models.CharField(choices=[("nutrition_plan", "Plan de Nutrición"), ("training_plan", "Plan de Entrenamiento"), ("medical_history", "Historia Clínica")], max_length=40)),
                ("file_name", models.CharField(max_length=255)),
                ("file_url", models.TextField()),
                ("extracted_text", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="documents", to="api.user")),
            ],
            options={
                "unique_together": {("user", "doc_type")},
            },
        ),
    ]
