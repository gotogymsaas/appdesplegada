from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0011_user_profession"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="full_name",
            field=models.CharField(blank=True, max_length=160, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="favorite_exercise_time",
            field=models.CharField(blank=True, max_length=80, null=True),
        ),
        migrations.AddField(
            model_name="user",
            name="favorite_sport",
            field=models.CharField(blank=True, max_length=120, null=True),
        ),
    ]
