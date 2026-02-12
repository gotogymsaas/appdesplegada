from rest_framework import serializers
from .models import User

class UserSerializer(serializers.ModelSerializer):
    profile_picture = serializers.SerializerMethodField()

    def get_profile_picture(self, obj):
        if not obj.profile_picture:
            return None
        try:
            url = obj.profile_picture.url
        except Exception:
            return None

        request = self.context.get("request")
        if request and url.startswith("/"):
            return request.build_absolute_uri(url)
        return url

    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'plan', 'age', 'weight', 'height', 'profession', 'full_name',
            'favorite_exercise_time', 'favorite_sport', 'happiness_index', 'scores', 'profile_picture',
            'current_streak', 'badges', 'date_joined', 'is_superuser', 'trial_active', 'trial_ends_at',
            'billing_status'
        ]

