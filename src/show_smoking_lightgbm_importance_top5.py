import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# LightGBM 라이브러리 import
try:
    import lightgbm as lgb
except ImportError:
    print("'lightgbm' 라이브러리가 설치되지 않았습니다. pip install lightgbm 명령어로 설치해주세요.")
    exit()

def visualize_smoking_lightgbm_top5(model_path, data_path, output_path):
    """
    저장된 흡연 LightGBM 모델을 불러와 상위 5개 변수 중요도를 시각화하고 저장합니다.
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

    print("상위 5개 변수 중요도 그래프를 생성합니다...")
    importances = model.feature_importances_
    normalized_importances = importances / importances.sum()
    feature_importances = pd.Series(normalized_importances, index=features).sort_values(ascending=False)

    top5_features = feature_importances.head(5)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top5_features, y=top5_features.index)
    plt.title('LightGBM 흡연 예측 모델: 상위 5개 변수 중요도', fontsize=14)
    plt.xlabel('중요도 (Importance)', fontsize=12)
    plt.ylabel('변수 (Feature)', fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"\n성공! 상위 5개 변수 중요도 그래프가 '{output_path}'에 저장되었습니다.")

if __name__ == '__main__':
    visualize_smoking_lightgbm_top5(
        model_path='results/smoking/smoking_model_lightgbm.joblib',
        data_path='analysis_data.csv',
        output_path='results/smoking/lightgbm_feature_importance_top5.png'
    )
