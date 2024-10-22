import pandas as pd
import seaborn as sns
import streamlit as st
import json
from catboost import CatBoostRegressor

PATH_DATA = 'spb.csv'
PATH_UNIQUE_VALUES = 'unique_values.json'
PATH_MODEL = 'models/catboost5.cbm'


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    return data


@st.cache_data
def load_model(path):
    model = CatBoostRegressor()
    model.load_model(path)
    return model


@st.cache_data
def transform(data):
    colors = sns.color_palette("coolwarm").as_hex()
    n_colors = len(colors)

    data = data.reset_index(drop=True)
    data["norm_price"] = data["price"] / data["area"]

    data["label_colors"] = pd.qcut(data["norm_price"], n_colors, labels=colors)
    data["label_colors"] = data["label_colors"].astype("str")
    return data


df = load_data(PATH_DATA)
df = transform(df)

st.markdown("""
    ## Предсказание цены на жильё в Санкт-Петербурге
    
    Карта с квартирами, данные о которых взяты за основу
""")

st.map(data=df, latitude='geo_lat', longitude='geo_lon', color='label_colors', size=10)

st.markdown(
    """
    ### Описание полей

        Тип здания: 
        1 - Панельный. 2 - Монолитный. 3 - Кирпичный. 4 - Блочный. 5 - Деревянный. 0 - Другое. \n
        Количество комнат: 
        Значение равное "0" означает квартиру-студию.\n
        Тип квартиры:
        1 - Вторичный рынок недвижимости. 11 - Новостройка.
        
    """
)

with open(PATH_UNIQUE_VALUES) as file:
    dict_unique = json.load(file)

geo_lat = st.sidebar.slider(
    "Координата широты", min_value=min(dict_unique['geo_lat']), max_value=max(dict_unique['geo_lat'])
)
geo_lon = st.sidebar.slider(
    "Координата долготы", min_value=min(dict_unique['geo_lon']), max_value=max(dict_unique['geo_lon'])
)
building_type = st.sidebar.selectbox('Тип здания', dict_unique['building_type'])
level = st.sidebar.slider(
    "Этаж квартиры", min_value=min(dict_unique['level']), max_value=max(dict_unique['level'])
)
levels = st.sidebar.slider(
    "Количество этажей в здании", min_value=min(dict_unique['levels']), max_value=max(dict_unique['levels'])
)
rooms = st.sidebar.selectbox('Количество комнат', dict_unique['rooms'])
area = st.sidebar.slider(
    "Общая площадь", min_value=min(dict_unique['area']), max_value=max(dict_unique['area'])
)
kitchen_area = st.sidebar.slider(
    "Площадь кухни", min_value=min(dict_unique['kitchen_area']), max_value=max(dict_unique['kitchen_area'])
)
object_type = st.sidebar.selectbox('Тип квартиры', dict_unique['object_type'])
year = st.sidebar.slider(
    "Год", min_value=min(dict_unique['year']), max_value=max(dict_unique['year'])
)
month = st.sidebar.slider(
    "Месяц", min_value=min(dict_unique['month']), max_value=max(dict_unique['month'])
)

dict_data = {
    'geo_lat': geo_lat,
    'geo_lon': geo_lon,
    'building_type': building_type,
    'level': level,
    'levels': levels,
    'rooms': rooms,
    'area': area,
    'kitchen_area': kitchen_area,
    'object_type': object_type,
    'year': year,
    'month': month,
    'level_to_levels': level / levels,
    'area_to_rooms': area / rooms if rooms != 0 else area
}

data_predict = pd.DataFrame([dict_data])
model = load_model(PATH_MODEL)

button = st.button("Рассчитать")

if button:
    output = model.predict(data_predict)[0]
    st.success(f"{round(output):,} рублей")
