from .utils import ModelLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# 모델 로딩
models = ModelLoader.load()
tfidf_w = models['tfidf_w']
# tfidf_c, ohe, scaler, kmeans도 필요시 활용 가능

# CSV 로드
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(THIS_DIR, 'model', 'tourist_spots.csv')
df = pd.read_csv(csv_path, encoding='utf-8')

# TF-IDF 벡터 계산 (word-level만)
W = tfidf_w.transform(df['캐치프레이즈']).toarray()

# 프로토타입 유사도 점수
prototypes = ["자연", "힐링", "산책", "휴식", "편안", "경사", "벤치", "전망", "체험", "조용"]
proto_tfs = tfidf_w.transform(prototypes).toarray()
proto_center = proto_tfs.mean(axis=0).reshape(1, -1)
senior_scores = cosine_similarity(W, proto_center).flatten()
senior_scores = (senior_scores + 1) / 2  # 정규화

# 추천 함수
def recommend(region, rejected=None, k=3):
    if rejected is None:
        rejected = set()
    df_r = df[df['지역 (시_군_구)'] == region]
    idxs = df_r.index.values
    if len(idxs) == 0:
        raise ValueError(f"'{region}'에 관광지가 없습니다.")
    
    recs, missing = [], []
    for c in range(3):
        cand = idxs[df_r['cluster'] == c]
        if len(cand) == 0:
            missing.append(c)
            continue
        logits = np.array([
            senior_scores[i] * (0.5 if df.at[i, '제목'] in rejected else 1)
            for i in cand
        ])
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        choice = np.random.choice(cand, p=probs)
        recs.append(choice)

    if missing:
        pool = [i for i in idxs if i not in recs]
        logits = np.array([
            senior_scores[i] * (0.5 if df.at[i, '제목'] in rejected else 1)
            for i in pool
        ])
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        extra = np.random.choice(pool, size=len(missing), replace=False, p=probs)
        recs.extend(extra)

    recs = list(dict.fromkeys(recs))
    return df.loc[recs, ['제목','cluster','캐치프레이즈']].to_dict(orient='records')
