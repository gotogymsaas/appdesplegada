from rest_framework import serializers
from .models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'plan', 'age', 'weight', 'height', 'profession', 'full_name',
            'favorite_exercise_time', 'favorite_sport', 'happiness_index', 'scores', 'profile_picture',
            'current_streak', 'badges', 'date_joined', 'is_superuser', 'trial_active', 'trial_ends_at',
            'billing_status'
        ]

