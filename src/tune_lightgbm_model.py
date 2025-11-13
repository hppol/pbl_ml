
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
import os

# LightGBM 라이브러리 import
try:
    import lightgbm as lgb
except ImportError:
    print("'lightgbm' 라이브러리가 설치되지 않았습니다. pip install lightgbm 명령어로 설치해주세요.")
    exit()

def tune_lightgbm_model():
    """
    LightGBM 모델을 훈련하고 GridSearchCV로 최적 하이퍼파라미터를 찾습니다.
    """
    print("--- LightGBM 모델 훈련 및 튜닝 시작 ---")

    # 1. 데이터 로드
    try:
        df = pd.read_csv('analysis_data.csv')
        print("1. 'analysis_data.csv' 파일 로드 완료.")
    except FileNotFoundError:
        print("오류: 'analysis_data.csv' 파일을 찾을 수 없습니다.")
        return

    # 2. 데이터 전처리
    print("2. 데이터 전처리 중...")
    df.fillna(df.median(), inplace=True)
    
    X = df.drop('BS3_1', axis=1)
    y = df['BS3_1'].apply(lambda x: 1 if x in [1, 2] else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("훈련/테스트 데이터 분할 완료.")

    # 3. scale_pos_weight 계산
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"3. 불균형 데이터 처리를 위한 scale_pos_weight 계산 완료: {scale_pos_weight:.2f}")

    # 4. GridSearchCV 설정
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50] # LightGBM의 주요 파라미터
    }

    lgbm_classifier = lgb.LGBMClassifier(
        objective='binary',
        metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    grid_search = GridSearchCV(lgbm_classifier, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
    print("\n4. GridSearchCV 탐색 시작... (XGBoost보다 빠를 수 있습니다)")
    grid_search.fit(X_train, y_train)
    print("탐색 완료.")

    # 5. 결과 출력
    print("\n--- 튜닝 결과 ---")
    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
    print(f"교차 검증 최고 점수 (f1_weighted): {grid_search.best_score_:.4f}")

    # 6. 최적 모델로 최종 평가
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"\n테스트 데이터 정확도: {accuracy_score(y_test, y_pred):.4f}")
    print("\n[LightGBM 최종 성능 평가 리포트]")
    print(classification_report(y_test, y_pred, target_names=['Non-Smoker', 'Smoker']))

    # 7. 최적 모델 저장
    print("7. 최적화된 LightGBM 모델 저장 중...")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'smoking_model_lightgbm.joblib')
    joblib.dump(best_model, model_path)
    print(f"모델이 '{model_path}' 파일로 저장되었습니다.")
    print("--- 모든 작업 완료 ---")

if __name__ == '__main__':
    tune_lightgbm_model()
