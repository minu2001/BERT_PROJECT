import pandas as pd
import ast
import matplotlib.pyplot as plt

# 📁 파일 경로 설정
overlap_path = "C:/Users/0120c/BERT_PROJECT/analysis/error_topic_overlap_with_keywords.csv"
examples_path = "C:/Users/0120c/BERT_PROJECT/analysis/topic_full/topic_examples.csv"
summary_path = "C:/Users/0120c/BERT_PROJECT/analysis/topic_full/topic_summary.csv"
output_csv = "C:/Users/0120c/BERT_PROJECT/analysis/matched_keyword_topic_ratio.csv"
output_png = "C:/Users/0120c/BERT_PROJECT/analysis/matched_keyword_topic_ratio.png"

# 📌 파일 로드
df_overlap = pd.read_csv(overlap_path, encoding="utf-8-sig")
df_examples = pd.read_csv(examples_path, encoding="utf-8-sig")
df_summary = pd.read_csv(summary_path, encoding="utf-8-sig")

# 🔧 키워드 리스트 파싱
df_examples["clean_keywords"] = df_examples["keywords"].apply(ast.literal_eval)
df_overlap["keywords_per_topic"] = df_overlap["keywords_per_topic"].apply(ast.literal_eval)

# 전체 문서 수
total_docs = df_summary["Count"].sum()

# 📊 결과 계산
result_rows = []

for idx, row in df_overlap.iterrows():
    model_pair = row["model_pair"]
    keywords_dict = row["keywords_per_topic"]

    all_keywords = set()
    for word_list in keywords_dict.values():
        all_keywords.update(word_list)

    example_grouped = df_examples.groupby("topic").first().reset_index()
    example_grouped["has_overlap"] = example_grouped["clean_keywords"].apply(
        lambda topic_kw: any(k in topic_kw for k in all_keywords)
    )
    matched_topics = example_grouped[example_grouped["has_overlap"]]["topic"].tolist()

    matched_count = df_summary[df_summary["Topic"].isin(matched_topics)]["Count"].sum()
    matched_ratio = round(matched_count / total_docs * 100, 2)

    result_rows.append({
        "model_pair": model_pair,
        "matched_topic_count": len(matched_topics),
        "matched_doc_count": matched_count,
        "matched_doc_ratio(%)": matched_ratio
    })

# 📄 DataFrame 생성 및 CSV 저장
df_result = pd.DataFrame(result_rows)
df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"✅ CSV 저장 완료: {output_csv}")

# 🎨 시각화
plt.figure(figsize=(10, 6))
bars = plt.bar(df_result["model_pair"], df_result["matched_doc_ratio(%)"], color="mediumseagreen")

# 라벨
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}%", ha='center', va='bottom', fontsize=10)

plt.title("모델별 공통 오답 키워드가 전체 문서에서 차지하는 비중", fontsize=14)
plt.xlabel("모델 쌍", fontsize=11)
plt.ylabel("Matched Document Ratio (%)", fontsize=11)
plt.xticks(rotation=45)
plt.ylim(0, df_result["matched_doc_ratio(%)"].max() + 10)
plt.tight_layout()

# 이미지 저장
plt.savefig(output_png, dpi=300)
print(f"🖼️ 시각화 이미지 저장 완료: {output_png}")

plt.show()
