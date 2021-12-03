
from folium.plugins import MarkerCluster
import folium
import pandas as pd
from Contact10 import tops, eda


def folium_map():
    patient_info, patient_route, region = eda()
    suspect = tops(patient_info["접촉자수"], 10)
    # 용의자들의 위도, 경도를 확인하기 위해 다른 데이터프레임과 병합후 필요한 정보만 가진 데이터프레임으로 생성
    suspect_route = pd.merge(suspect, patient_route[["환자ID", "latitude", "longitude"]], on=['환자ID'])
    suspect_route = suspect_route[["환자ID", "시도", "접촉자수", "latitude", "longitude"]]
    suspect_final = suspect_route.groupby(["환자ID"])["latitude", "longitude"].mean()
    suspect_x = suspect.set_index("환자ID")

    # 용의자 10명중 위도, 경도가 없는 경우가 있어서 "region"에서 지역별 확진자의 위도 , 경도를 추출
    region_mean = region.groupby("지역").mean()

    # 용의자 10명 위도,경도가 없는 경우 용의자들의 "시도"정보를 가지고 위도, 경도 값 추가
    # 용의자 10명의 환자 ID와  용의자 10명중 위도, 경도값이 있는 사람들의 환자 ID를 리스트
    suspect_list = suspect_x.index.tolist()
    fin_list = suspect_final.index.tolist()

    # 환자ID을 차집합으로 없는 것만 골라내서 suspect_final 데이터프레임에 행 추가
    for i in list(set(suspect_list) - set(fin_list)):
        for j in region_mean.index:
            if suspect_x.loc[i, "시도"] == j:
                suspect_final.loc[i] = region_mean.loc[j][["latitude", "longitude"]]

    fin_fin = suspect_final.reset_index("환자ID")
    fin_df = pd.merge(fin_fin, suspect[["환자ID", "시도", "접촉자수", ]], on=['환자ID']).set_index("환자ID")

    # 용의자 10명의 위도 경도 좌표를 이용해 folium을 사용해 지도에  노출
    map = folium.Map(location=[36.55, 127], tiles="cartodbpositron", zoom_start=7, width='60%', height='70%')

    marker_cluster = MarkerCluster().add_to(map)

    for a in fin_df.index:
        folium.Marker(location=[fin_df.loc[a, "latitude"], fin_df.loc[a, "longitude"]],
                      # zoom_start=12,
                      tooltip=fin_df.loc[a, "시도"] + ":" + str(fin_df.loc[a, "접촉자수"]) + "명",
                      popup=(fin_df.loc[a, "시도"])).add_to(marker_cluster)

    map.save("/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/static/image/folium_map.html")



