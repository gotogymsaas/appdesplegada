from rest_framework import serializers
from .models import DeviceConnection


class DeviceConnectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceConnection
        fields = ["provider", "status", "last_sync_at", "updated_at"]

