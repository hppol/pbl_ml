
import pandas as pd

def print_csv_columns(filepath):
    """
    CSV 파일의 헤더만 빠르게 읽어 모든 컬럼명을 출력합니다.
    """
    try:
        # nrows=0으로 데이터를 읽지 않고 헤더만 읽어옵니다.
        df_header = pd.read_csv(filepath, nrows=0)
        columns = df_header.columns.tolist()
        
        print(f"'{filepath}' 파일의 전체 변수(컬럼) 목록입니다.")
        print(f"총 {len(columns)}개의 변수가 있습니다.\n")
        
        # 보기 편하도록 한 줄에 5개씩 출력
        for i in range(0, len(columns), 5):
            print(", ".join(columns[i:i+5]))
            
    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    print_csv_columns('combined_output.csv')
