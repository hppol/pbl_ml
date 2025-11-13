
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os

def tune_hyperparameters():
    """
    GridSearchCV를 사용하여 랜덤 포레스트 모델의 최적 하이퍼파라미터를 찾고,
    개선된 모델을 저장합니다.
    """
    print("--- 하이퍼파라미터 튜닝 시작 ---")

    # 1. 데이터 로드 (가벼운 analysis_data.csv 사용)
    try:
        df = pd.read_csv('analysis_data.csv')
        print("1. 'analysis_data.csv' 파일 로드 완료.")
    except FileNotFoundError:
        print("오류: 'analysis_data.csv' 파일을 찾을 수 없습니다. 먼저 create_subset_csv.py를 실행하세요.")
        return

    # 2. 데이터 전처리
    print("2. 데이터 전처리 중...")
    df.fillna(df.median(), inplace=True)
    
    X = df.drop('BS3_1', axis=1)
    y = df['BS3_1'].apply(lambda x: 1 if x in [1, 2] else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("훈련/테스트 데이터 분할 완료.")

    # 3. 파이프라인 및 하이퍼파라미터 그리드 설정
    # SMOTE와 RandomForest를 파이프라인으로 묶어 교차 검증 시 데이터 유출 방지
    pipeline = ImblearnPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # 테스트할 하이퍼파라미터 그리드 (범위를 줄여 시간 단축)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20],
        'classifier__min_samples_leaf': [2, 4],
        'classifier__class_weight': ['balanced', None]
    }
    print("3. 하이퍼파라미터 탐색 그리드 설정 완료.")

    # 4. GridSearchCV 실행
    # f1_weighted: 불균형 데이터셋에서 recall과 precision을 모두 고려하는 평가지표
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
    print("\n4. GridSearchCV 탐색 시작... (시간이 다소 소요될 수 있습니다)")
    grid_search.fit(X_train, y_train)
    print("탐색 완료.")

    # 5. 결과 출력
    print("\n--- 튜닝 결과 ---")
    print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
    print(f"교차 검증 최고 점수 (f1_weighted): {grid_search.best_score_:.4f}")

    # 6. 최적 모델로 최종 평가
    print("\n6. 최적 모델로 테스트 데이터 평가 중...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"\n테스트 데이터 정확도: {accuracy_score(y_test, y_pred):.4f}")
    print("\n[최종 성능 평가 리포트]")
    print(classification_report(y_test, y_pred, target_names=['Non-Smoker', 'Smoker']))

    # 7. 최적 모델 저장
    print("7. 최적화된 모델 저장 중...")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'smoking_model_tuned.joblib')
    joblib.dump(best_model, model_path)
    print(f"모델이 '{model_path}' 파일로 저장되었습니다.")
    print("--- 모든 작업 완료 ---")

if __name__ == '__main__':
    tune_hyperparameters()
