from django.urls import path
from .views import RecommendAPIView

urlpatterns = [
    path('recommend/', RecommendAPIView.as_view(), name='recommend'),
]