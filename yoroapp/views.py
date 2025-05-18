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

class FeedbackAPIView(APIView):
    """
    POST body 예시:
    {
      "rejected_titles": ["제목A", "제목B", "제목C"]
    }
    """
    def post(self, request):
        titles = request.data.get('rejected_titles')
        if not isinstance(titles, list):
            return Response({'error': 'rejected_titles를 리스트로 보내주세요.'},
                            status=status.HTTP_400_BAD_REQUEST)

        # ❷ 세션에 기존 rejected_titles가 있으면 합치고, 없으면 새로 만든 뒤 저장
        prev = set(request.session.get('rejected_titles', []))
        prev.update(titles)
        request.session['rejected_titles'] = list(prev)

        return Response({'status': 'ok'}, status=status.HTTP_200_OK)
    

class FeedbackTestPage(APIView):
    def get(self, request):
        return render(request, 'feedback_test.html')