import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# 경로 추가 (패키지 인식 문제 방지용)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모델별 예측 함수 import
# ✅ 수정된 코드 (정상 작동)
from models.MobileBERT.MobileBERT_Inference import predict_sentiment as predict_mobile
from models.FinBERT.FinBERT_Inference import predict_sentiment as predict_fin
from models.DeBERTa.DeBERTa_Inference import predict_sentiment as predict_deb


# 📌 데이터 로드 및 전처리
print("📌 [1] 데이터 로드: Sentiment_Stock_data_30k.csv")
df = pd.read_csv("data_file/Sentiment_Stock_data_30k.csv", encoding="cp949", encoding_errors="ignore")
df["labels"] = df["labels"].astype(int)
texts = df["sentence"].astype(str).tolist()

# 📌 예측 실행 (0,1,2)
print("🚀 예측 시작")
df["MobileBERT"] = predict_mobile(texts)
df["FinBERT"] = predict_fin(texts)
df["DeBERTa"] = predict_deb(texts)

# 📁 예측 결과 저장
output_path = "data_file/stock_30k_full_predictions.csv"
df.to_csv(output_path, index=False, encoding="cp949")
print(f"✅ 저장 완료: {output_path}")


# ✅ 중립(2) 제외하여 binary 평가 (0,1)
def evaluate_binary(name, true, pred):
    df_bin = pd.DataFrame({"true": true, "pred": pred})
    df_bin = df_bin[df_bin["pred"] != 2].reset_index(drop=True)
    excluded = len(true) - len(df_bin)

    print(f"\n📊 {name} (중립 제외)")
    print(f"Neutral 제외: {excluded}개 ({excluded / len(true):.2%})")
    print(f"Accuracy (0/1): {accuracy_score(df_bin['true'], df_bin['pred']):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(df_bin["true"], df_bin["pred"], labels=[0, 1]))
    print("Classification Report:\n",
          classification_report(df_bin["true"], df_bin["pred"], labels=[0, 1], zero_division=0))


# 🔍 평가 실행
evaluate_binary("MobileBERT", df["labels"], df["MobileBERT"])
evaluate_binary("FinBERT", df["labels"], df["FinBERT"])
evaluate_binary("DeBERTa", df["labels"], df["DeBERTa"])

# 📊 중립 예측 수 시각화
neutral_counts = {
    "MobileBERT": (df["MobileBERT"] == 2).sum(),
    "FinBERT": (df["FinBERT"] == 2).sum(),
    "DeBERTa": (df["DeBERTa"] == 2).sum()
}
sns.barplot(x=list(neutral_counts.keys()), y=list(neutral_counts.values()))
plt.title("모델별 중립(2) 예측 개수")
plt.ylabel("개수")
plt.show()
