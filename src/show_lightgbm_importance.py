
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

def visualize_lightgbm_importance(model_path, data_path, output_path):
    """
    저장된 LightGBM 모델을 불러와 변수 중요도를 시각화하고 저장합니다.
    """
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    print(f"'{model_path}'에서 LightGBM 모델을 불러옵니다...")
    try:
        model = joblib.load(model_path)
        print("모델 로드 완료.")
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        return

    print(f"'{data_path}'에서 변수명을 가져옵니다...")
    try:
        df = pd.read_csv(data_path, nrows=0)
        features = df.drop('BS3_1', axis=1).columns.tolist()
    except (FileNotFoundError, KeyError):
        print(f"오류: '{data_path}' 파일 또는 'BS3_1' 컬럼을 찾을 수 없습니다.")
        return

    print("변수 중요도 그래프를 생성합니다...")
    # LightGBM의 feature_importances_는 분할에 사용된 횟수이므로, 합으로 나누어 정규화
    importances = model.feature_importances_
    normalized_importances = importances / importances.sum()
    
    feature_importances = pd.Series(normalized_importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('LightGBM 흡연 예측 모델의 변수 중요도')
    plt.xlabel('중요도 (Importance)')
    plt.ylabel('변수 (Feature)')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"\n성공! 변수 중요도 그래프가 '{output_path}'에 저장되었습니다.")
    print("\n[LightGBM 모델 변수 중요도 TOP 5]")
    print(feature_importances.head(5))

if __name__ == '__main__':
    visualize_lightgbm_importance(
        model_path='results/smoking_model_lightgbm.joblib',
        data_path='analysis_data.csv',
        output_path='results/lightgbm_feature_importance.png'
    )
