import numpy as np
import pandas as pd
import os
from .utils import ModelLoader

# 모델 로딩
models = ModelLoader.load()
tfidf_w = models['tfidf_vectorizer_w']

# CSV 로드
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(THIS_DIR, 'model', 'tourist_spots.csv')
df = pd.read_csv(csv_path, encoding='utf-8')

# NumPy 기반 cosine similarity 함수
def cosine_sim_matrix(X, Y):
    # X: (n_samples, n_features), Y: (m_samples, n_features)
    # 반환: (n_samples, m_samples)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)  # (n,1)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)  # (m,1)
    sim = X @ Y.T / (X_norm * Y_norm.T + 1e-9)
    return sim

def recommend(region, rejected=None, k=3):
    if rejected is None:
        rejected = set()

    # ① TF-IDF 벡터 계산
    W = tfidf_w.transform(df['캐치프레이즈']).toarray()

    # ② 프로토타입 유사도 계산 (NumPy 전용)
    prototypes = ["자연","힐링","산책","휴식","편안","경사","벤치","전망","체험","조용"]
    proto_tfs    = tfidf_w.transform(prototypes).toarray()
    proto_center = proto_tfs.mean(axis=0).reshape(1, -1)  # (1, n_features)
    # cosine_sim_matrix에선 Y가 (1, n_features)이므로 반환 shape은 (n_samples,1)
    senior_scores = cosine_sim_matrix(W, proto_center).flatten()
    senior_scores = (senior_scores + 1) / 2  # [0,1] 정규화

    # ③ 지역 필터링
    df_r = df[df['지역 (시_군_구)'] == region]
    idxs = df_r.index.values
    if len(idxs) == 0:
        raise ValueError(f"'{region}'에 관광지가 없습니다.")

    # ④ 클러스터별 softmax 샘플링
    recs, missing = [], []
    for c in range(3):
        cand = idxs[df_r['cluster'] == c]
        if not len(cand):
            missing.append(c)
            continue
        logits = np.array([
            senior_scores[i] * (0.5 if df.at[i,'제목'] in rejected else 1)
            for i in cand
        ])
        exp   = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        recs.append(np.random.choice(cand, p=probs))

    # ⑤ 부족 클러스터 보충
    if missing:
        pool   = [i for i in idxs if i not in recs]
        logits = np.array([
            senior_scores[i] * (0.5 if df.at[i,'제목'] in rejected else 1)
            for i in pool
        ])
        exp   = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        extra = np.random.choice(pool, size=len(missing), replace=False, p=probs)
        recs.extend(extra)

    # ⑥ 중복 제거 & 결과 반환
    recs = list(dict.fromkeys(recs))
    return df.loc[recs, ['제목','cluster','캐치프레이즈']].to_dict(orient='records')
