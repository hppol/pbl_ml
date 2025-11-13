import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# LightGBM 라이브러리 import
try:
    import lightgbm as lgb
except ImportError:
    print("'lightgbm' 라이브러리가 설치되지 않았습니다. pip install lightgbm 명령어로 설치해주세요.")
    exit()

def visualize_lightgbm_importance_pie(model_path, data_path, output_path, top_n=10):
    """
    저장된 LightGBM 모델을 불러와 변수 중요도를 원형 그래프로 시각화하고 저장합니다.
    """
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    print(f"'{model_path}'에서 LightGBM 모델을 불러옵니다...")
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

    print("변수 중요도 원형 그래프를 생성합니다...")
    importances = model.feature_importances_
    normalized_importances = importances / importances.sum()
    feature_importances = pd.Series(normalized_importances, index=features).sort_values(ascending=False)

    # 상위 N개 변수 선택 및 나머지 '기타'로 그룹화
    top_features = feature_importances.head(top_n)
    if len(feature_importances) > top_n:
        other_importance = feature_importances.iloc[top_n:].sum()
        pie_data = pd.concat([top_features, pd.Series({'기타': other_importance})])
    else:
        pie_data = top_features

    plt.figure(figsize=(12, 12))
    wedges, texts, autotexts = plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
    plt.setp(autotexts, size=8, weight="bold", color="white")
    plt.title(f'LightGBM 흡연 예측 모델 변수 중요도 (상위 {top_n}개)', fontsize=16)
    plt.axis('equal')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"\n성공! 변수 중요도 원형 그래프가 '{output_path}'에 저장되었습니다.")

if __name__ == '__main__':
    visualize_lightgbm_importance_pie(
        model_path='results/smoking/smoking_model_lightgbm.joblib',
        data_path='analysis_data.csv',
        output_path='results/smoking/lightgbm_feature_importance_pie.png'
    )
