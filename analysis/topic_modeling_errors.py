import pandas as pd
import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

def analyze_model_lda(model_name, input_csv, text_column, n_topics=5, output_base="analysis/lda_compare"):
    print(f"🚀 {model_name} LDA 분석 시작...")
    os.makedirs(f"{output_base}/{model_name}", exist_ok=True)

    df = pd.read_csv(input_csv, encoding="cp949", encoding_errors="ignore")
    texts = df[text_column].dropna().astype(str).tolist()

    vectorizer = CountVectorizer(stop_words="english", max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    doc_topic = lda.fit_transform(X)
    topic_assignments = doc_topic.argmax(axis=1)

    # 🔹 정량분석: 토픽별 문서 수
    topic_counts = Counter(topic_assignments)

    # 🔸 정성분석: 토픽별 키워드
    feature_names = vectorizer.get_feature_names_out()
    keyword_rows = []
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        keyword_rows.append({"topic": f"Topic {idx}", "keywords": ", ".join(top_words)})
    pd.DataFrame(keyword_rows).to_csv(f"{output_base}/{model_name}/topic_keywords.csv", index=False, encoding="utf-8-sig")

    # 🔸 정성분석: 토픽별 대표 문장
    df["assigned_topic"] = topic_assignments
    example_rows = []
    for topic_id in range(n_topics):
        topic_df = df[df["assigned_topic"] == topic_id].head(2)
        for i, row in topic_df.iterrows():
            example_rows.append({
                "topic": f"Topic {topic_id}",
                "rank": i + 1,
                "sentence": row[text_column]
            })
    pd.DataFrame(example_rows).to_csv(f"{output_base}/{model_name}/topic_examples.csv", index=False, encoding="utf-8-sig")

    print(f"✅ {model_name}: 키워드 및 대표 문장 저장 완료")
    return [topic_counts.get(i, 0) for i in range(n_topics)]  # 토픽별 문서 수 리스트

# 🔁 모델별 실행
model_files = {
    "MobileBERT": "C:/Users/0120c/BERT_PROJECT/data_file/error_mobilebert.csv",
    "FinBERT": "C:/Users/0120c/BERT_PROJECT/data_file/error_deberta.csv",
    "DeBERTa": "C:/Users/0120c/BERT_PROJECT/data_file/error_finbert.csv"
}

topic_counts_all = {}
n_topics = 5

for model, path in model_files.items():
    topic_counts_all[model] = analyze_model_lda(model, path, text_column="sentence", n_topics=n_topics)

# 📊 시각화: 모델별 오답 토픽 분포
x = range(n_topics)
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar([i - width for i in x], topic_counts_all["MobileBERT"], width=width, label="MobileBERT")
plt.bar(x, topic_counts_all["FinBERT"], width=width, label="FinBERT")
plt.bar([i + width for i in x], topic_counts_all["DeBERTa"], width=width, label="DeBERTa")

plt.xticks(x, [f"T{i+1}" for i in x])
plt.xlabel("Topic")
plt.ylabel("Document Count")
plt.title("LDA 기반 모델별 오답 토픽 분포 비교")
plt.legend()
plt.tight_layout()

os.makedirs("lda_compare", exist_ok=True)
plt.savefig("analysis/lda_compare/lda_topic_distribution_comparison.png")
plt.show()

print("🎉 전체 모델 LDA 정량+정성 분석 완료!")
