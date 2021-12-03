import warnings
from pyecharts.charts import Geo
from pyecharts.globals import CurrentConfig, NotebookType, ChartType, SymbolType
from pyecharts import options as opts
import pandas as pd
import pandas as pd
import numpy as np
from view_eda import *
criminal_list , patient_info, region = j_eda()
# 누가 범인 인가
def who_is_criminal(criminal_list):
    global patient_info
    # 1차 숙주에게 감염당한 2차 숙주
    infected_patient = []

    for criminal in criminal_list:
        for i in patient_info[patient_info['infected_by'] == criminal]['patient_id'].values:
            infected_patient.append(i)

    return infected_patient


# 범인 정보 조회
def criminal_info(patient_info_, criminal_list):
    criminal_patient_info = pd.DataFrame(columns=patient_info_.columns)
    for criminal in criminal_list:
        criminal_patient_info = criminal_patient_info.append(patient_info_[patient_info_['patient_id'] == criminal])

    return criminal_patient_info


def where_is_criminal(criminal_list, region, patient_info_, html_name):
    criminal = criminal_info(patient_info_, criminal_list)
    criminal_route = pd.merge(criminal, region, on='place')
    p_id = criminal_route['patient_id']
    city = criminal_route['place']
    longi = criminal_route['longitude']  # 위도
    lati = criminal_route['latitude']  # 경도
    data = list(zip(p_id, city, longi, lati))

    geo = Geo(
        init_opts=opts.InitOpts(width="600px",
                                height="600px",
                                page_title="시각화",
                                bg_color="#404a59")
    )
    geo.add_schema(
        maptype="韩国",  # 한국
        itemstyle_opts=opts.ItemStyleOpts(
            color="#323c48",  # background color
            border_color="white"  # Boundary line color
        )
    )

    for p_id, city, longi, lati in data:
        geo.add_coordinate("장소", latitude=lati, longitude=longi)  #
        geo.add(series_name='장소',  # Series name
                data_pair=[("장소", f'{city}')],
                type_='effectScatter',
                symbol_size=20,  # 동그라미 크기
                color='pink',
                is_selected=True
                )

        geo.set_series_opts(label_opts=opts.LabelOpts(
            is_show=False
        )
        )

    geo.set_global_opts(
        title_opts=opts.TitleOpts(
            title='',  #
            # subtitle = f'기간 : {start_date} ~ {end_date}',#
            item_gap=15,  #
            title_textstyle_opts=opts.TextStyleOpts(
                color="white",  # color
                font_weight="bolder",  #
                font_size=40  #
            ),
            subtitle_textstyle_opts=opts.TextStyleOpts(
                color='white',  # color
                font_weight="bolder",  #
                font_size=15)
        ),
        legend_opts=opts.LegendOpts(
            pos_right="10px",
            inactive_color="white",
            textstyle_opts=opts.TextStyleOpts(color="orange")  # 장소 하고 o 표시 설명 박스
        )
    )

    geo.render(f"/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/static/image/{html_name}.html")


def patient_spread(start_criminal_list, end_criminal_list, region, patient_info_, html_name):
    # 확진자 정보 조회
    start_criminal = criminal_info(patient_info_, start_criminal_list)
    end_criminal = criminal_info(patient_info_, end_criminal_list)
    start_criminal_route = pd.merge(start_criminal, region, on='place')
    end_criminal_route = pd.merge(end_criminal, region, on='place')

    # start_criminal_route 전처리
    start_criminal_route.drop(columns=['infected_by'], axis=1, inplace=True)
    start_criminal_route.rename(columns={'patient_id': 'infected_by'}, inplace=True)
    start_criminal_route = start_criminal_route.reset_index()
    # start_point 찍기
    start_criminal_route['start_point'] = start_criminal_route['index'].map(lambda x: 'start' + str(x))

    # end_point 찍기
    end_df = pd.DataFrame(end_criminal_route.groupby(['latitude', 'longitude'])['patient_id'].count()).reset_index()
    end_df = end_df.reset_index()
    end_df['end_point'] = end_df['index'].map(lambda x: 'end' + str(x))
    end_df.drop(columns=['patient_id'], axis=1, inplace=True)

    # start to end DataFrame
    route = pd.merge(end_criminal_route, end_df, on=['latitude', 'longitude'])
    route = pd.merge(route, start_criminal_route, on='infected_by')

    # draw graph
    p_id = route['patient_id']
    start_city = route['place_y']
    start_point = route['start_point']
    start_longi = route['longitude_y']
    start_lati = route['latitude_y']
    start_data = list(zip(start_city, start_point, start_longi, start_lati))

    end_city = route['place_x']
    end_point = route['end_point']
    end_longi = route['longitude_x']
    end_lati = route['latitude_x']
    end_data = list(zip(end_city, end_point, end_longi, end_lati))

    geo = Geo(
        init_opts=opts.InitOpts(width="600px",
                                height="600px",
                                page_title="시각화",
                                bg_color="#404a59")
    )
    geo.add_schema(
        maptype="韩国",  # 한국
        itemstyle_opts=opts.ItemStyleOpts(
            color="#323c48",  # background color
            border_color="white")  # Boundary line color
    )

    for s_city, s_point, s_longi, s_lati, in start_data:
        geo.add_coordinate(name=f"{s_point}",
                           longitude=s_longi,
                           latitude=s_lati)

        geo.add(series_name=f"",  # Series name
                data_pair=[(f"{s_point}", f'{s_city}')],
                type_=ChartType.EFFECT_SCATTER,
                symbol_size=20,  # 동그라미 크기
                color='skyblue',
                is_selected=True
                )

    for e_city, e_point, e_longi, e_lati, in end_data:
        geo.add_coordinate(name=f"{e_point}",
                           longitude=e_longi,
                           latitude=e_lati)

        geo.add(series_name=f"",  # Series name
                data_pair=[(f"{e_point}", f'{e_city}')],
                type_=ChartType.EFFECT_SCATTER,
                symbol_size=10,  # 동그라미 크기
                color=['skyblue', 'red', 'yellow'],
                is_selected=True
                )

    geo.add("geo-lines",
            list(zip(start_point, end_point)),
            type_=ChartType.LINES,
            effect_opts=opts.EffectOpts(symbol=SymbolType.ARROW,
                                        symbol_size=10,
                                        color="white"),
            is_polyline=False,
            linestyle_opts=opts.LineStyleOpts(curve=0.2),
            color='yellow',
            is_large=True)

    geo.set_series_opts(
        label_opts=opts.LabelOpts(is_show=None))

    geo.set_global_opts(
        title_opts=opts.TitleOpts(
            title='',
            subtitle=f'',
            item_gap=15,
            title_textstyle_opts=opts.TextStyleOpts(
                color="white",
                font_weight="bolder",
                font_size=60
            ),
            subtitle_textstyle_opts=opts.TextStyleOpts(
                color='red',  # color
                font_weight="bolder",  #
                font_size=20)
        )
    )
    geo.render(f"/home/ubuntu/ai/cloud/workspace/flask_project/hello_flask/static/image/{html_name}.html")