
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_tuned_importance(model_path, data_path, output_path):
    """
    저장된 튜닝 모델을 불러와 변수 중요도를 시각화하고 저장합니다.
    """
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    print(f"'{model_path}'에서 튜닝된 모델을 불러옵니다...")
    try:
        pipeline = joblib.load(model_path)
        model = pipeline.named_steps['classifier']
        print("모델 로드 완료.")
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        return
    except KeyError:
        print("오류: 불러온 모델이 예상한 파이프라인 구조가 아닙니다.")
        return

    print(f"'{data_path}'에서 변수명을 가져옵니다...")
    try:
        df = pd.read_csv(data_path, nrows=0)
        features = df.drop('BS3_1', axis=1).columns.tolist()
    except (FileNotFoundError, KeyError):
        print(f"오류: '{data_path}' 파일 또는 'BS3_1' 컬럼을 찾을 수 없습니다.")
        return

    print("변수 중요도 그래프를 생성합니다...")
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('튜닝된 흡연 예측 모델의 변수 중요도')
    plt.xlabel('중요도 (Importance)')
    plt.ylabel('변수 (Feature)')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"\n성공! 변수 중요도 그래프가 '{output_path}'에 저장되었습니다.")
    print("\n[신뢰도 높은 변수 중요도 TOP 5]")
    print(feature_importances.head(5))

if __name__ == '__main__':
    visualize_tuned_importance(
        model_path='results/smoking_model_tuned.joblib',
        data_path='analysis_data.csv',
        output_path='results/tuned_feature_importance.png'
    )

