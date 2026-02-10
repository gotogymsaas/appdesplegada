from django.contrib import admin
from .models import User, ContactMessage, PushToken, TermsAcceptance

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('id', 'username', 'email', 'plan', 'is_staff', 'is_active')
    search_fields = ('username', 'email')


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email', 'subject', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('name', 'email', 'subject')


@admin.register(PushToken)
class PushTokenAdmin(admin.ModelAdmin):
    list_display = ('id', 'platform', 'user', 'active', 'last_seen_at', 'created_at')
    list_filter = ('platform', 'active', 'created_at')
    search_fields = ('token', 'device_id', 'user__username', 'user__email')


@admin.register(TermsAcceptance)
class TermsAcceptanceAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'version', 'accepted_at', 'source')
    list_filter = ('version', 'source', 'accepted_at')
    search_fields = ('user__username', 'user__email', 'version')
