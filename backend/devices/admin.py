from django.contrib import admin
from django.contrib import admin
from .models import DeviceConnection

@admin.register(DeviceConnection)
class DeviceConnectionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "provider", "status", "last_sync_at", "updated_at")
    list_filter = ("provider", "status")
    search_fields = ("user__username", "user__email")

