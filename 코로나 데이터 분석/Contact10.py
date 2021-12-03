import pandas as pd
from flask import Flask
from waitress import serve
import numpy as np
import matplotlib.pyplot as plt
from flask import render_template
from datetime import datetime
from folium.plugins import MarkerCluster
import folium

patient_info_path = '/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/data/PatientInfo.csv'
patient_route_path = '/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/data/PatientRoute.csv'
region_path = '/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/data/Region.csv'

region = pd.read_csv(region_path, encoding="UTF8")
patient_info = pd.read_csv(patient_info_path, encoding="UTF8")
patient_route = pd.read_csv(patient_route_path, encoding="cp949")


def eda():
    patient_info.rename(columns={
                                  'patient_id':'환자ID',
                                  "sex":"성별",
                                  "birth_year":"출생년도",
                                  "age":"연령대",
                                  "province":"시도",
                                  "city":"시군구",
                                  "infection_case" : "감염경로",
                                  "infection_order" : "n차 감염",
                                  "infected_by" : "감염시킨ID",
                                  "contact_number" : "접촉자수",
                                  "confirmed_date" : "확진일",
                                  "released_date"  : "격리해제일",
                                  "deceased_date"  : "사망날짜"
                                  },inplace=True)
    patient_route.rename(columns={
                                  'patient_id':'환자ID',
                                  "date" : "다녀간날짜",
                                  "province" : "시도",
                                  "city"  : "시군구",
                                 "type"  : "종류",

                                  },inplace=True)

    region.rename(columns={
                        "province":'지역',
                       'university_count':'대학수'
                        }, inplace = True)

    return patient_info, patient_route, region
patient_info, patient_route, region = eda()
#########################################################################################
def tops(col_value,n):

  df = pd.DataFrame(columns= patient_info.columns)
  topx_list = [x for x in col_value.tolist() if pd.isnull(x) == False]
  topx = topx_list.sort(reverse=True)
  topx = topx_list[0:n]
  for i in topx:
    df = df.append(patient_info[col_value == i])
  return df

suspect = tops(patient_info["접촉자수"],10)

