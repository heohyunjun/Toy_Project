import bar_chart_race as bcr
from Contact10 import tops,fun_map,eda

def race_chart():
    patient_info, patient_route, region = eda()
    region_corfiremed = patient_info.groupby(['확진일', '시도'])['환자ID'].count().reset_index()
    region_corfiremed = region_corfiremed.rename(columns={"환자ID": "확진자수"})

    df = region_corfiremed.pivot_table(
        index=['확진일'],
        columns=['시도'],
        values='확진자수'
    )

    df.fillna(0, inplace=True)  # 널값 0으로 변경
    df = df.sort_values(list(df.columns)).sort_index()  # 정렬

    df.iloc[:, 0:-1] = df.iloc[:, 0:-1].cumsum()
    top_clubs = set()
    for index, row in df.iterrows():
        top_clubs = set(row[row > 0].sort_values(ascending=False).head(10).index)

    df = df[top_clubs]

    bcr.bar_chart_race(df=df,
                       n_bars=10,
                       period_length=170,
                       steps_per_period=10,
                       filename='race.mp4',
                       sort='desc',
                       title='confirm_race'
                       )
