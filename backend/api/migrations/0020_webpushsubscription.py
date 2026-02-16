from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0019_user_terms_accepted_at_user_terms_accepted_ip_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='WebPushSubscription',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('endpoint', models.TextField(unique=True)),
                ('p256dh', models.CharField(max_length=255)),
                ('auth', models.CharField(max_length=255)),
                ('device_id', models.CharField(blank=True, default='', max_length=120)),
                ('active', models.BooleanField(default=True)),
                ('last_seen_at', models.DateTimeField(auto_now=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='web_push_subscriptions', to='api.user')),
            ],
        ),
    ]
