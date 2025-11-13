import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os

# imbalanced-learn 라이브러리 import
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("'imbalanced-learn' 라이브러리가 설치되지 않았습니다. pip install imbalanced-learn 명령어로 설치해주세요.")
    exit()

# 경고 메시지 무시
warnings.filterwarnings('ignore')
# 한글 폰트 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def create_smoking_prediction_model(df):
    """
    흡연 예측 모델을 훈련하고 저장하는 함수
    """
    model_name = 'smoking_prediction'
    target_col = 'BS3_1'
    features = [
        'age', 'sex', 'HE_dbp', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_BMI', 
        'BP16_1', 'BP16_2', 'HE_sbp', 'D_1_1'
    ]
    define_target_lambda = lambda x: 1 if x in [1, 2] else 0

    print(f"--- {model_name} 모델 훈련 시작 ---")

    # 1. 데이터 전처리
    print("\n1. 데이터 전처리 중...")
    required_columns = features + [target_col]
    if not all(col in df.columns for col in required_columns):
        print(f"오류: 필수 컬럼이 CSV 파일에 없습니다.")
        return

    df_analysis = df[required_columns].copy()
    df_analysis['target'] = df_analysis[target_col].apply(define_target_lambda)
    df_analysis = df_analysis.drop(target_col, axis=1)
    
    df_analysis.fillna(df_analysis.median(), inplace=True)
    print("결측치 처리 및 목표 변수 생성이 완료되었습니다.")

    # 2. 데이터 분할, SMOTE, 훈련
    X = df_analysis.drop('target', axis=1)
    y = df_analysis['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    # 3. 평가
    print("\n3. 모델 성능 평가 중...")
    y_pred = model.predict(X_test)
    print(f"모델 정확도: {accuracy_score(y_test, y_pred):.4f}")
    print("\n[성능 평가 리포트]")
    print(classification_report(y_test, y_pred, target_names=['Non-Smoker', 'Smoker']))

    # 4. 저장
    print("\n4. 모델과 변수 중요도 그래프 저장 중...")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, f'{model_name}.joblib'))
    
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title(f'{model_name} 모델의 변수 중요도')
    plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'))
    plt.close()
    print("모델과 그래프가 'results' 폴더에 저장되었습니다.")
    print(f"--- {model_name} 모델 훈련 완료 ---\\n")

if __name__ == '__main__':
    try:
        df_main = pd.read_csv('combined_output.csv', low_memory=False)
        print(f"데이터 로드 완료: {df_main.shape[0]} 행, {df_main.shape[1]} 열")
        create_smoking_prediction_model(df_main)
    except FileNotFoundError:
        print("오류: 'combined_output.csv' 파일을 찾을 수 없습니다.")
