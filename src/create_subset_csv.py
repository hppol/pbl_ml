import pandas as pd

def create_subset_csv(input_filepath, output_filepath, columns_to_keep):
    """
    큰 CSV 파일에서 특정 컬럼만 선택하여 새로운 CSV 파일을 만듭니다.
    """
    print(f"'{input_filepath}' 파일을 읽는 중입니다...")
    try:
        # 필요한 컬럼만 읽어 메모리 효율성 증대
        df = pd.read_csv(input_filepath, usecols=lambda c: c in columns_to_keep, low_memory=False)
        print("필요한 변수만 선택하여 파일을 읽었습니다.")

        # 순서를 정의한대로 고정
        df = df[columns_to_keep]

        df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        print(f"성공! '{output_filepath}' 파일이 생성되었습니다.")
        print(f"포함된 변수 ({len(columns_to_keep)}개): {columns_to_keep}")

    except FileNotFoundError:
        print(f"오류: '{input_filepath}' 파일을 찾을 수 없습니다.")
    except ValueError as e:
        print(f"오류: 지정한 변수 중 일부가 원본 CSV 파일에 없는 것 같습니다. - {e}")
    except Exception as e:
        print(f"작업 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    # 흡연/음주 예측에 필요한 모든 변수 목록
    required_vars = [
        # Features (19)
        'age', 'sex', 'HE_dbp', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_BMI', 
        'BP16_1', 'BP16_2', 'HE_sbp', 'D_1_1',
        'mh_stress', 'BP_PHQ_9', 'BP_GAD_7', 'dr_month', 'BD1_11', 
        'edu', 'occp', 'ho_incm',

        # Target for Smoking (1)
        'BS3_1',
        # Target for Drinking (1)
        'BD2_31'
    ]
    
    create_subset_csv(
        input_filepath='combined_output.csv',
        output_filepath='analysis_data.csv',
        columns_to_keep=required_vars
    )