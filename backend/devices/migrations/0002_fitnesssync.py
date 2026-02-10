from django.db import migrations, models
import django.db.models.deletion
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        ('devices', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='FitnessSync',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('provider', models.CharField(max_length=40)),
                ('start_time', models.DateTimeField()),
                ('end_time', models.DateTimeField()),
                ('metrics', models.JSONField(blank=True, default=dict)),
                ('raw', models.JSONField(blank=True, default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='fitness_syncs', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'indexes': [
                    models.Index(fields=['user', 'provider', 'created_at'], name='devices_fit_user_pr_1964ea_idx'),
                ],
            },
        ),
    ]
