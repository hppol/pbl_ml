
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
import os

# XGBoost 라이브러리 import
try:
    import xgboost as xgb
except ImportError:
    print("'xgboost' 라이브러리가 설치되지 않았습니다. pip install xgboost 명령어로 설치해주세요.")
    exit()

def tune_drinking_xgboost_model():
    """
    XGBoost를 사용하여 고위험 음주 예측 모델을 훈련하고 튜닝합니다.
    """
    print("--- 고위험 음주 예측 XGBoost 모델 훈련 및 튜닝 시작 ---")

    # 1. 데이터 로드
    try:
        df = pd.read_csv('analysis_data.csv')
    except FileNotFoundError:
        print("오류: 'analysis_data.csv' 파일을 찾을 수 없습니다.")
        return

    # 2. 데이터 전처리 및 목표 변수 정의
    df.fillna(df.median(), inplace=True)
    df['target'] = df['BD2_31'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    X = df.drop(['BS3_1', 'BD2_31', 'target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. scale_pos_weight 계산
    try:
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    except ZeroDivisionError:
        print("오류: 훈련 데이터에 고위험 음주(클래스 1) 샘플이 없습니다.")
        return

    # 4. GridSearchCV 설정
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    grid_search = GridSearchCV(xgb_classifier, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # 5. 결과 출력 및 저장
    print("\n--- 튜닝 결과 ---")
    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("\n[XGBoost 최종 성능 평가 리포트]")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'High-Risk']))

    output_dir = "results/drinking"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'drinking_model_xgboost.joblib')
    joblib.dump(best_model, model_path)
    print(f"\n모델이 '{model_path}' 파일로 저장되었습니다.")

if __name__ == '__main__':
    tune_drinking_xgboost_model()
