import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# XGBoost 라이브러리 import
try:
    import xgboost as xgb
except ImportError:
    print("'xgboost' 라이브러리가 설치되지 않았습니다. pip install xgboost 명령어로 설치해주세요.")
    exit()

def visualize_drinking_xgboost_importance(model_path, data_path, output_path):
    """
    저장된 고위험 음주 XGBoost 모델을 불러와 변수 중요도를 시각화하고 저장합니다.
    """
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    print(f"'{model_path}'에서 모델을 불러옵니다...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        return

    print(f"'{data_path}'에서 변수명을 가져옵니다...")
    try:
        df = pd.read_csv(data_path, nrows=0)
        features = df.drop(['BS3_1', 'BD2_31'], axis=1).columns.tolist()
    except (FileNotFoundError, KeyError) as e:
        print(f"오류: 데이터 파일 또는 필요 컬럼을 찾을 수 없습니다. {e}")
        return

    print("변수 중요도 그래프를 생성합니다...")
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    num_features = len(feature_importances)
    plt.figure(figsize=(12, max(10, num_features * 0.4)))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('XGBoost 고위험 음주 예측 모델의 변수 중요도')
    plt.xlabel('중요도 (Importance)')
    plt.ylabel('변수 (Feature)')
    plt.yticks(fontsize=8)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"\n성공! 변수 중요도 그래프가 '{output_path}'에 저장되었습니다.")
    print("\n[XGBoost 고위험 음주 모델 변수 중요도 TOP 5]")
    print(feature_importances.head(5))

if __name__ == '__main__':
    visualize_drinking_xgboost_importance(
        model_path='results/drinking/drinking_model_xgboost.joblib',
        data_path='analysis_data.csv',
        output_path='results/drinking/xgboost_drinking_importance_horizontal.png'
    )
