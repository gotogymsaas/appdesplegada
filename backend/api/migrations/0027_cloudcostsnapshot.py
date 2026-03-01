from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0026_merge_0023_intelligence_campaigns_0025_user_sex_field"),
    ]

    operations = [
        migrations.CreateModel(
            name="CloudCostSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("scope", models.CharField(max_length=255)),
                ("date_from", models.DateField()),
                ("date_to", models.DateField()),
                ("timezone", models.CharField(max_length=64)),
                ("source", models.CharField(default="", max_length=64)),
                ("actual_cost_usd", models.FloatField(default=0.0)),
                ("actual_cost_cop", models.FloatField(default=0.0)),
                ("estimated_cost_usd", models.FloatField(default=0.0)),
                ("estimated_cost_cop", models.FloatField(default=0.0)),
                ("estimation_error_pct", models.FloatField(blank=True, null=True)),
                ("lag_hours", models.FloatField(default=0.0)),
                ("by_service", models.JSONField(blank=True, default=list)),
                ("by_resource_group", models.JSONField(blank=True, default=list)),
                ("extra_meta", models.JSONField(blank=True, default=dict)),
                ("refreshed_at", models.DateTimeField(auto_now=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "unique_together": {("scope", "date_from", "date_to", "timezone")},
                "indexes": [
                    models.Index(fields=["scope", "date_from", "date_to", "timezone"], name="api_cloudco_scope_f5056e_idx"),
                    models.Index(fields=["refreshed_at"], name="api_cloudco_refresh_77a3d6_idx"),
                ],
            },
        ),
    ]
