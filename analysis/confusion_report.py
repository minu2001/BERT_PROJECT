import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.ticker as mticker

# 📁 데이터 로드
df = pd.read_csv(
    "C:/Users/0120c/BERT_PROJECT/data_file/stock_30k_full_predictions.csv",
    encoding="cp949",
    encoding_errors="ignore"
)

# 📁 저장 디렉토리 확인/생성
os.makedirs("analysis", exist_ok=True)

# 📊 Confusion Matrix + Classification Report 시각화 함수
def plot_confusion_and_report(model_name):
    # 모델이 예측한 값이 0 또는 1인 데이터만 사용 (중립 예측 제외)
    df_model = df[df[model_name] != 2]

    # 정답 라벨은 0, 1, 2 중에 뭐든 포함될 수 있으므로 유지
    y_true = df_model["labels"]
    y_pred = df_model[model_name]

    # 혼동 행렬과 리포트 계산
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)

    # 🔹 Confusion Matrix 시각화
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"analysis/confusion_matrix_{model_name}.png")
    plt.close()

    # 🔹 Classification Report 시각화
    report_df = pd.DataFrame(report).transpose().iloc[:2][["precision", "recall", "f1-score"]]
    report_df.plot(kind="bar", figsize=(8, 5))
    plt.title(f"Classification Report: {model_name}")
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"analysis/classification_report_{model_name}.png")
    plt.close()

    # 🔸 요약 출력
    acc = report["accuracy"]
    print(f"✅ {model_name}: Accuracy={acc:.4f}, Samples used={len(df_model)}")
    print(f"📊 confusion_matrix_{model_name}.png / classification_report_{model_name}.png 저장 완료\n")

# 🔁 모델별 실행
plot_confusion_and_report("MobileBERT")
plot_confusion_and_report("FinBERT")
plot_confusion_and_report("DeBERTa")
