import pandas as pd
import os
import json

# ✅ 절대경로로 지정
keyword_files = {
    "MobileBERT": r"C:\Users\0120c\BERT_PROJECT\analysis\analysis\lda_compare\MobileBERT\topic_keywords.csv",
    "FinBERT": r"C:\Users\0120c\BERT_PROJECT\analysis\analysis\lda_compare\FinBERT\topic_keywords.csv",
    "DeBERTa": r"C:\Users\0120c\BERT_PROJECT\analysis\analysis\lda_compare\DeBERTa\topic_keywords.csv",
}

# 키워드 및 토픽셋 저장용
topic_sets = {}
topic_keywords_dict = {}

# 🔹 모델별 topic set과 keyword dict 생성
for model, path in keyword_files.items():
    if not os.path.exists(path):
        print(f"❌ 파일 없음: {path}")
        continue
    df = pd.read_csv(path, encoding="utf-8-sig")

    topic_sets[model] = set(df["topic"])
    topic_keywords_dict[model] = {
        row["topic"]: row["keywords"].split(", ") if isinstance(row["keywords"], str) else []
        for _, row in df.iterrows()
    }

# ✅ 교집합 분석 + 키워드 매핑
all_models = list(topic_sets.keys())
overlap_result = []

for i in range(len(all_models)):
    for j in range(i + 1, len(all_models)):
        m1, m2 = all_models[i], all_models[j]
        common_topics = topic_sets[m1].intersection(topic_sets[m2])

        # 키워드 병합: topic -> list of keywords (중복 제거)
        keywords_per_topic = {}
        for topic in sorted(common_topics):
            k1 = topic_keywords_dict[m1].get(topic, [])
            k2 = topic_keywords_dict[m2].get(topic, [])
            merged_keywords = sorted(set(k1 + k2))
            keywords_per_topic[str(topic)] = merged_keywords  # key를 문자열로 저장

        overlap_result.append({
            "model_pair": f"{m1} & {m2}",
            "overlap_count": len(common_topics),
            "overlapping_topics": sorted(common_topics),
            "keywords_per_topic": json.dumps(keywords_per_topic, ensure_ascii=False)  # JSON 문자열로 저장
        })

# ✅ 저장
df_overlap = pd.DataFrame(overlap_result)
save_path = r"/analysis/error_topic_overlap_with_keywords.csv"
df_overlap.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"🎉 교차 분석 + 키워드 저장 완료: {save_path}")
