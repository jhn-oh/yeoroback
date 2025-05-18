from django.urls import path
from .views import RecommendAPIView, FeedbackAPIView, FeedbackTestPage
from django.shortcuts import render


urlpatterns = [
    path('recommend/', RecommendAPIView.as_view(), name='recommend'),
    path('feedback/', FeedbackAPIView.as_view(), name='feedback'),
    path('test/page/', FeedbackTestPage.as_view(), name='feedback_test'),
    path('test/recommend/', RecommendAPIView.as_view()),
    path('test/feedback/', FeedbackAPIView.as_view()),

]