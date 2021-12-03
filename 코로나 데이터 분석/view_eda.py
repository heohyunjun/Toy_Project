import pandas as pd
import numpy as np

patient_info_path = '/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/data/PatientInfo.csv'
region_path = '/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/data/Region.csv'
patient_info = pd.read_csv(patient_info_path, encoding="UTF8")
region = pd.read_csv(region_path, encoding="UTF8")

def j_eda():
    patient_info_path = '/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/data/PatientInfo.csv'
    region_path = '/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/data/Region.csv'
    patient_info = pd.read_csv(patient_info_path, encoding="UTF8")
    region = pd.read_csv(region_path, encoding="UTF8")

    patient_info['city'] = patient_info['city'].replace({'etc': np.nan})
    patient_info['city'] = patient_info['city'].fillna(patient_info['province'])
    # 확진일 결측치 제거
    patient_info.dropna(subset=['confirmed_date'], axis=0, inplace=True)
    # place 칼럼 추가
    patient_info['place'] = patient_info['province'] + ' ' + patient_info['city']
    # 필요없는 columns 제거
    patient_info = patient_info.drop(['global_num', 'country', 'province', 'city', 'symptom_onset_date'],
                                     axis=1).sort_values('confirmed_date')
    # 날짜 형식 통일
    patient_info["confirmed_date"] = pd.to_datetime(patient_info['confirmed_date'])
    # place 칼럼 추가
    region['place'] = region['province'] + ' ' + region['city']

    # 환자별 확진 시킨 빈도수 조사
    infected_patient = pd.DataFrame(patient_info.groupby(['infected_by'])['patient_id'].count())
    # 확진시킨 빈도 Top 10
    most_infected = sorted(infected_patient['patient_id'].values, reverse=True)[:10]
    # Top 10의 정보 DataFrame 생성
    criminal_patient_info = pd.DataFrame(columns=patient_info.columns)

    for i in set(most_infected):
        for p_id in infected_patient[infected_patient['patient_id'] == i].index:
            criminal_patient_info = criminal_patient_info.append(patient_info[patient_info['patient_id'] == p_id])

    # 확진 빈도 컬럼 추가
    criminal_patient_info['infect_count'] = most_infected
    # 범인 list에 담기
    criminal_list = criminal_patient_info['patient_id'].values

    return criminal_list , patient_info, region

