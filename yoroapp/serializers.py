# recommender/serializers.py
from rest_framework import serializers

class SpotSerializer(serializers.Serializer):
    제목 = serializers.CharField()
    cluster = serializers.IntegerField()
    캐치프레이즈 = serializers.CharField()
