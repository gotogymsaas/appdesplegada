from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone

class User(AbstractUser):
    plan = models.CharField(
        max_length=20,
        choices=[("Gratis", "Gratis"), ("Premium", "Premium")],
        default="Gratis"
    )
    age = models.PositiveSmallIntegerField(null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    profession = models.CharField(max_length=120, null=True, blank=True)
    full_name = models.CharField(max_length=160, null=True, blank=True)
    favorite_exercise_time = models.CharField(max_length=80, null=True, blank=True)
    favorite_sport = models.CharField(max_length=120, null=True, blank=True)
    happiness_index = models.FloatField(null=True, blank=True)
    scores = models.JSONField(default=dict, blank=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', null=True, blank=True)
    current_streak = models.IntegerField(default=0)
    last_streak_date = models.DateField(null=True, blank=True)
    badges = models.JSONField(default=list, blank=True)
    trial_active = models.BooleanField(default=False)
    trial_started_at = models.DateTimeField(null=True, blank=True)
    trial_ends_at = models.DateTimeField(null=True, blank=True)
    billing_status = models.CharField(max_length=20, default="free")
    subscription_provider = models.CharField(max_length=30, null=True, blank=True)
    subscription_id = models.CharField(max_length=120, null=True, blank=True)
    current_period_end = models.DateTimeField(null=True, blank=True)
    cancel_at_period_end = models.BooleanField(default=False)
    last_payment_status = models.CharField(max_length=20, null=True, blank=True)
    terms_accepted_at = models.DateTimeField(null=True, blank=True)
    terms_accepted_version = models.CharField(max_length=20, blank=True, default="")
    terms_accepted_ip = models.GenericIPAddressField(null=True, blank=True)
    terms_accepted_user_agent = models.TextField(blank=True, default="")
    terms_accepted_source = models.CharField(max_length=30, blank=True, default="web")

    # Sync & coaching state (para experiencia tipo "Laura")
    timezone = models.CharField(max_length=64, blank=True, default="")
    coach_state = models.JSONField(default=dict, blank=True)
    coach_state_updated_at = models.DateTimeField(null=True, blank=True)
    coach_weekly_state = models.JSONField(default=dict, blank=True)
    coach_weekly_updated_at = models.DateTimeField(null=True, blank=True)


    def __str__(self):
        return self.username

class HappinessRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='history')
    value = models.FloatField()
    scores = models.JSONField(default=dict, blank=True)
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.value} ({self.date})"


class IFQuestion(models.Model):
    key = models.CharField(max_length=64, unique=True)
    label = models.CharField(max_length=255)
    order = models.PositiveSmallIntegerField(default=0)
    active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.order}. {self.key}"


class IFAnswer(models.Model):
    SLOT_CHOICES = [
        ("morning", "morning"),
        ("afternoon", "afternoon"),
        ("night", "night"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='if_answers')
    question = models.ForeignKey(IFQuestion, on_delete=models.CASCADE, related_name='answers')
    week_id = models.CharField(max_length=10)
    slot = models.CharField(max_length=12, choices=SLOT_CHOICES, null=True, blank=True)
    value = models.PositiveSmallIntegerField()
    source = models.CharField(max_length=20, default="app")
    created_at = models.DateTimeField(auto_now_add=True)
    answered_at = models.DateTimeField(auto_now=True)
    answered_date = models.DateField(default=timezone.localdate)

    class Meta:
        unique_together = ("user", "question", "week_id", "answered_date", "slot")

    def __str__(self):
        return f"{self.user.username} - {self.question.key} ({self.week_id})"


class UserDocument(models.Model):
    DOC_TYPES = [
        ("nutrition_plan", "Plan de Nutrición"),
        ("training_plan", "Plan de Entrenamiento"),
        ("medical_history", "Historia Clínica"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="documents")
    doc_type = models.CharField(max_length=40, choices=DOC_TYPES)
    file_name = models.CharField(max_length=255)
    file_url = models.TextField()
    extracted_text = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "doc_type")

    def __str__(self):
        return f"{self.user.username} - {self.doc_type}"


class ContactMessage(models.Model):
    STATUS_CHOICES = [
        ("received", "Recibido"),
        ("responded", "Respondido"),
    ]

    name = models.CharField(max_length=160)
    email = models.EmailField()
    subject = models.CharField(max_length=200)
    message = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="received")
    created_at = models.DateTimeField(auto_now_add=True)
    responded_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.name} - {self.subject}"


class PushToken(models.Model):
    PLATFORM_CHOICES = [
        ("android", "Android"),
        ("ios", "iOS"),
        ("web", "Web"),
        ("unknown", "Unknown"),
    ]

    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name="push_tokens")
    token = models.CharField(max_length=512, unique=True)
    platform = models.CharField(max_length=20, choices=PLATFORM_CHOICES, default="unknown")
    device_id = models.CharField(max_length=120, blank=True, default="")
    active = models.BooleanField(default=True)
    last_seen_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.platform}:{self.token[:12]}..."


class WebPushSubscription(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name="web_push_subscriptions")
    endpoint = models.TextField(unique=True)
    p256dh = models.CharField(max_length=255)
    auth = models.CharField(max_length=255)
    device_id = models.CharField(max_length=120, blank=True, default="")
    active = models.BooleanField(default=True)
    last_seen_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"web:{self.endpoint[:24]}..."


class TermsAcceptance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="terms_acceptances")
    version = models.CharField(max_length=20)
    accepted_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True, default="")
    source = models.CharField(max_length=30, default="web")

    class Meta:
        indexes = [
            models.Index(fields=["user", "version"]),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.version} ({self.accepted_at})"


class AuditLog(models.Model):
    actor = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="audit_logs",
    )
    occurred_at = models.DateTimeField(auto_now_add=True)

    action = models.CharField(max_length=80)
    entity_type = models.CharField(max_length=40, blank=True, default="")
    entity_id = models.CharField(max_length=120, blank=True, default="")

    before_json = models.JSONField(null=True, blank=True)
    after_json = models.JSONField(null=True, blank=True)

    reason = models.TextField(blank=True, default="")
    ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True, default="")

    class Meta:
        indexes = [
            models.Index(fields=["occurred_at"]),
            models.Index(fields=["action"]),
            models.Index(fields=["entity_type", "entity_id"]),
        ]

    def __str__(self):
        return f"{self.occurred_at} {self.action} {self.entity_type}:{self.entity_id}"

