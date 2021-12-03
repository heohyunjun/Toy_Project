"""한글 꺠짐 잡아야함"""

import matplotlib.pyplot as plt
import pandas as pd
from Contact10 import tops,eda
from datetime import datetime
import seaborn as sns
def trend_chart(html_name):
    plt.rc('font', family = "NanumGothic")
    patient_info, patient_route, region = eda()

    # patioent['확진일'] 열의 널값 행 삭제
    patient_info.dropna(subset=["확진일"], inplace=True)
    # 확진일 타입을 datetime으로 변경
    patient_info["확진일"] = pd.to_datetime(patient_info["확진일"])
    # 확진일 기준으로 환자ID 수 count
    day_patient = patient_info.groupby(["확진일"])["환자ID"].count()

    # 확진자 추세, 정책 라인 적용
    plt.figure(figsize=(10,5)) # 사이즈조절
    sns.lineplot(data = day_patient , x=day_patient.index, y=day_patient.values) # 선그래프
    plt.axvline(x=datetime(2020, 2, 22), color='r', linestyle='--', linewidth=3,label="2020, 2, 22") # 첫 거리두기 시행
    plt.axvline(x=datetime(2020, 3, 2), color='g', linestyle='--', linewidth=3,label="2020, 3, 2") # 등교 중지 발표
    plt.savefig(f"/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/static/image/{html_name}.png")