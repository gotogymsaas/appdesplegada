from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0010_happinessrecord_scores"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="profession",
            field=models.CharField(blank=True, max_length=120, null=True),
        ),
    ]
