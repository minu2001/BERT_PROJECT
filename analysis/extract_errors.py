import pandas as pd
import os

# 📌 파일 로드
file_path = os.path.join("C:/Users/0120c/BERT_PROJECT/data_file/stock_30k_full_predictions.csv")
df = pd.read_csv(file_path, encoding="cp949", encoding_errors="ignore")



# 🔍 오류 데이터 추출 함수
def extract_errors(model_name: str):
    # 예측과 정답이 다른 경우만 필터
    error_df = df[df["labels"] != df[model_name]].copy()

    # 불필요한 열 제거
    keep_cols = ["sentence", "labels", model_name]
    error_df = error_df[keep_cols]

    # 저장
    save_path = os.path.join("C:/Users/0120c/BERT_PROJECT/data_file", f"error_{model_name.lower()}.csv")
    error_df.to_csv(save_path, index=False, encoding="cp949")
    print(f"✅ {model_name} 오류 데이터 저장 완료: {save_path} ({len(error_df)} rows)")

# 모델별 실행
extract_errors("MobileBERT")
extract_errors("FinBERT")
extract_errors("DeBERTa")
