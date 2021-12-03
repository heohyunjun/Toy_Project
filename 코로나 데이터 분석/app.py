
"""# pdf 4페이지 // 코드 및 주석 """

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
from flask import Flask
from waitress import serve
import numpy as np
import matplotlib.pyplot as plt
from flask import render_template
from datetime import datetime
from folium.plugins import MarkerCluster
import folium
from Contact10 import tops,eda
from Patient_Trend import trend_chart
from Folium_Map import folium_map
from Academy import  academy_region_num
import warnings
from view_corona import *
from view_eda import j_eda
warnings.filterwarnings(action='ignore')


app = Flask(__name__)

# @app.route('/')
# def hello():
#     return 'hellow'

# @app.route("/corona22")
# def corona2():
#     folium_map()
#     return render_template('weather2.html', graph_file_name='folium_map.html', title="접촉자수 많은 사람 지역 분포")


@app.route("/start")
def trend_chart1():
   # trend_chart("trend")
   # academy_region_num("academy_region_num")
   # folium_map()  # folium_map
   # criminal_list, patient_info, region = j_eda()
   # where_is_criminal(criminal_list, region, patient_info, '1차분포')
   # where_is_criminal(who_is_criminal(criminal_list), region, patient_info, '2차분포')
   # patient_spread(criminal_list, who_is_criminal(criminal_list), region, patient_info, '1차확산')
   # patient_spread(who_is_criminal(criminal_list), who_is_criminal(who_is_criminal(criminal_list)), region,
   #                patient_info, '2차확산')
    return render_template('pro.html',
                           image_name1  = "trend.png",
                           image_name2 = "academy_region_num.png",

                           graph_file_name1 = "folium_map.html",
                           graph_file_name2="1차분포.html",
                           graph_file_name3="2차분포.html",
                           graph_file_name4="1차확산.html",
                           graph_file_name5="2차확산.html",

                           title="확진자 추세")

# @app.route("/")
# @app.route("/academy")
# def academy_num_region():
#     academy_region_num()
#     return render_template('weather1.html', image_name="academy.png", title="지역별 고등, 대학교 수")



# @app.route("/view_where_corona1")
# @app.route("/")
# def view_where_corona1():
#     criminal_list ,patient_info, region= j_eda()
#     where_is_criminal(criminal_list, region, patient_info, '1차분포')
#     return render_template('weather2.html', graph_file_name="1차분포.html", title="코로나 1차 분포")

# @app.route("/view_where_corona2")
# @app.route("/")
# def view_where_corona2():
#     criminal_list ,patient_info, region = j_eda()
#     where_is_criminal(who_is_criminal(criminal_list), region, patient_info, '2차분포')
#     return render_template('weather2.html', graph_file_name="2차분포.html", title="코로나 2차 분포")
# #
# # @app.route("/view_spread_corona1")
# @app.route("/")
# def view_spread_corona1():
#     criminal_list ,patient_info, region = j_eda()
#     patient_spread(criminal_list, who_is_criminal(criminal_list), region, patient_info, '1차확산')
#     return render_template('weather2.html', graph_file_name="1차확산.html", title="코로나 1차 확산")
#
# # @app.route("/view_spread_corona2")
# @app.route("/")
# def view_spread_corona2():
#     criminal_list , patient_info, region = j_eda()
#     patient_spread(who_is_criminal(criminal_list), who_is_criminal(who_is_criminal(criminal_list)), region, patient_info, '2차확산')
#     return render_template('weather2.html', graph_file_name="2차확산.html", title="코로나 2차 확산")

serve(app, host = '0.0.0.0', port = 8080)
