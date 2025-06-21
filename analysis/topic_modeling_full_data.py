import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import os

# 📁 설정
input_path = "C:/Users/0120c/BERT_PROJECT/data_file/stock_30k_full_predictions.csv"
output_dir = "topic_full"
os.makedirs(output_dir, exist_ok=True)

# 📌 데이터 로드 및 전처리
df = pd.read_csv(input_path, encoding="cp949", encoding_errors="ignore")
docs = df["sentence"].dropna().drop_duplicates().tolist()
print(f"✅ 전체 문장 수: {len(docs)}")

# 📌 CountVectorizer 설정
vectorizer_model = CountVectorizer(stop_words="english", max_df=0.95, min_df=10)

# 📌 BERTopic 모델 생성
topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    language="english",
    calculate_probabilities=False,
    verbose=True
)

# 📌 토픽 모델링 실행
topics, probs = topic_model.fit_transform(docs)

# 📁 문장별 결과 저장
df_result = pd.DataFrame({"sentence": docs, "topic": topics})
df_result.to_csv(f"{output_dir}/topic_full_data.csv", index=False, encoding="utf-8-sig")

# 📊 토픽 정보 요약 저장
topic_info = topic_model.get_topic_info()
topic_info.to_csv(f"{output_dir}/topic_summary.csv", index=False, encoding="utf-8-sig")

# 📝 정성 분석: 토픽별 대표 문장 2개씩 추출
examples = []
for topic_num in topic_info["Topic"]:
    if topic_num == -1:
        continue  # -1은 outlier 토픽 → 제외
    example_docs = df_result[df_result["topic"] == topic_num].head(2)["sentence"].tolist()
    keywords_only = [word for word, _ in topic_model.get_topic(topic_num)]  # ✅ 여기만 수정
    for i, sent in enumerate(example_docs):
        examples.append({
            "topic": topic_num,
            "rank": i + 1,
            "keywords": keywords_only,
            "example_sentence": sent
        })
example_df = pd.DataFrame(examples)
example_df.to_csv(f"{output_dir}/topic_examples.csv", index=False, encoding="utf-8-sig")

# 📊 시각화 저장
topic_model.visualize_barchart(top_n_topics=10).write_html(f"{output_dir}/top10_barchart.html")
topic_model.visualize_topics().write_html(f"{output_dir}/topic_clusters.html")
topic_model.visualize_heatmap().write_html(f"{output_dir}/topic_heatmap.html")

print("🎉 전체 문장 토픽모델링 + 정성분석 완료!")
print(f"📂 결과 저장 위치: {output_dir}")
