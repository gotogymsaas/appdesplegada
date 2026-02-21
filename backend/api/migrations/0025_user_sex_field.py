from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0024_user_goal_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="sex",
            field=models.CharField(
                max_length=10,
                choices=[("male", "male"), ("female", "female")],
                null=True,
                blank=True,
            ),
        ),
    ]
