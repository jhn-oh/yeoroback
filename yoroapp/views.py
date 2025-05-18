from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .recommender import recommend
from .serializers import SpotSerializer

class RecommendAPIView(APIView):
    def get(self, request):
        region = request.query_params.get('region')
        rejected = request.query_params.getlist('rejected', [])
        if not region:
            return Response({'error': 'region 파라미터가 필요합니다!'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            recs = recommend(region, set(rejected))
            serializer = SpotSerializer(recs, many=True)
            return Response(serializer.data)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_404_NOT_FOUND)
