#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import pickle
import numpy as np
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer

# data directory
DATA_hmda_acs = 'data/hmda_acs_merged.csv'
DATA_zip_tract = 'data/zip_tract_122017.xlsx'
DATA_shp = 'data/2019/tl_2019_53_tract.shp'
DATA_zipcodes = 'data/zipcodes_king.csv'
MODEL_lr_nh = 'data/lr_model_nh.sav'
MODEL_lr_hf = 'data/lr_model_hf.sav'

# dictionary of models
model_dict = {'Original': MODEL_lr_nh,
              'Without Population Bias': MODEL_lr_hf
              }

# transform the feature vectors
transformer = FunctionTransformer(np.log1p, validate=True)
scaler = MinMaxScaler(feature_range=(0.2, 0.8))

# set the title of web app
st.title('intelliRefinder')
st.markdown(
    '''Predict optimal locations for mortgage refinance business opportunities
       using machine learning algorithms on US OpenStreetMap (OSM) data.
    ''')

# load zip codes of king county WA
zipcodes = pd.read_csv(DATA_zipcodes)['zip']
#algorithms = ('Logistic Regression', 'Random Forest')
interventions = ('Original', 'Without Population Bias')

# seting up the sidebar and loading the data
st.sidebar.title('''Select your location''')
st.sidebar.markdown('''King County, WA''')
zipcode = st.sidebar.selectbox('Select zip code', zipcodes)
#algo = st.sidebar.selectbox('Select algorithm', algorithms)
st.sidebar.markdown('''Feature selection choice''')
itvn = st.sidebar.selectbox('Make a selection',interventions)
# Approach
st.sidebar.markdown('''**Approach**''')
st.sidebar.markdown(
    '''* Train logistic regression model to classfy mortgage refinance business \
    opportunities based on census tracts.''')
st.sidebar.markdown(
    '''* Implement the model to predict opportunies and visualize them in an \
    interactive maps for mortgage lenders.''')
st.sidebar.markdown(
    '''* Mortgage lenders actively locate refinance business opportunities \
    and tailor their resources towards those markets.''')
# Data
st.sidebar.markdown('''**Data**''')
st.sidebar.markdown(
    '''* 2008-2017 mortgage transaction data, available via Home Mortgage \
    Disclosure Act''')
st.sidebar.markdown(
    '''* Demographic data from 10-year Census American Community Survey data\
    ''')
st.sidebar.markdown('''* King County, WA selected for the demo project''')

# finding the census tract that associated with zipcode
@st.cache(persist=True, suppress_st_warning=True)
def zip2tract(state_code=530330):
    z2t = {}
    zt = []
    df = pd.read_excel(DATA_zip_tract)
    df.columnns = ['zip', 'tract']
    for z, t in zip(df['zip'], df['tract']):
        if str(t)[:6] == str(state_code):
            zt.append([z, str(t)[-5:].lstrip('0')])
        else:
            continue
    zt = pd.DataFrame(zt, columns=['zip', 'tract'])
    # if you tract number is 5 digits, you don't need to /100
    # however, you might want to change the tract from str to int
    zt['tract'] = zt['tract'].astype(float)/100
    z_unique =  zt['zip'].unique()
    for _, row in zt.iterrows():
        ts = zt[zt['zip'] == row[0]]
        z2t.update({row[0]: ts['tract']})
    return z2t

# reading shapefile and convert it based on census tracts
@st.cache(persist=True, suppress_st_warning=True)
def get_geodata(shp):
    gdf = gpd.read_file(shp).to_crs({'init':'epsg:4326'})
    gdf['TRACTCE'] = gdf['TRACTCE'].astype(float)/100
    gdf.rename(columns={'TRACTCE': 'census_tract'}, inplace=True)
    gdf['census_tract'] = gdf['census_tract'].astype(float)
    gdf = gdf[gdf['COUNTYFP'] == '033']
    return gdf

def get_tract(zipcode):
    zipcode = int(zipcode)
    return zip2tract()[zipcode]

# loading the merged data of hmda and acs
@st.cache(persist=True, suppress_st_warning=True)
def load_data(itvn):
    df = pd.read_csv(DATA_hmda_acs)
    ethical_related = df.columns[41:80]
    if itvn == 'Original':
        df.drop(
            ['loan_purpose_1', 'loan_purpose_2', 'loan_purpose_3'],
            axis=1,
            inplace=True
        )
    elif itvn == 'Without Population Bias':
        df.drop(ethical_related, axis=1, inplace=True)
    df.rename(columns={'Unnamed: 0': 'census_tract_number'}, inplace=True)
    return df

def select_model(itvn):
    return model_dict[itvn]

def load_model(itvn):
    model_path = select_model(itvn)
    return pickle.load(open(model_path, 'rb'))

def predict(tracts, itvn):
    df = load_data(itvn)
    df_x = df.iloc[:, :-2]
    idxs = df_x['census_tract_number'].isin(tracts.values)
    tracts = df_x[idxs].iloc[:, 0]
    df_x = df_x[idxs].iloc[:, 1:]
    x = transformer.transform(df_x)
    model = load_model(itvn)
    results = [max(i, j) for i, j in model.predict_proba(x)]
    tracts = np.asarray(tracts)
    results_merged = pd.DataFrame({'tracts': tracts, 'results': results})
    results_merged.rename(
        columns={
            'tracts':'census_tract_number',
            'results':'Refinance_score'},
        inplace=True
    )
    return results_merged

def map_plot(geo_data, data):
    lats = geo_data['INTPTLAT'].astype(float)
    lons = geo_data['INTPTLON'].astype(float)
    lat = lats.mean()
    lon = lons.mean()
    map = folium.Map(
        location=[lat, lon],
        zoom_start=11,
        control_scale=True,
        prefer_canvas=True,
        disable_3d=True)
    # add score layer
    score_layer = folium.FeatureGroup(name='Opportunity score')
    map.add_child(score_layer)
    # add choropleth layer
    folium.Choropleth(
        geo_data=geo_data,
        name='Census tracts',
        data=data,
        columns=['census_tract_number', 'Refinance_score'],
        key_on='feature.properties.census_tract',
        fill_color='YlGnBu',
        legend_name='Mortgage refinance business opportunity',
        na_fill_color='white',
        na_fill_opacity=0.2,
        fill_opacity=0.7,
        line_weight=0.6,
        line_opacity=0.2
    ).add_to(map)
    # add the markers
    for i in range(0, geo_data.shape[0]):
        lat_pop = lats.iloc[i]
        lon_pop = lons.iloc[i]
        tract = geo_data['census_tract'].iloc[i]
        score = data[data['census_tract_number'] == tract]['Refinance_score']
        folium.Marker(
            [lat_pop, lon_pop],
            popup='Score: {}'.format(round(float(score.values), 2))
        ).add_to(score_layer)
    # add layer control functions
    folium.LayerControl().add_to(map)
    return map

def main():
    tracts = get_tract(zipcode)
    scores = predict(tracts, itvn)
    gdf = get_geodata(DATA_shp)
    gdf_tract_set = set(gdf['census_tract'])
    scores_tract_set = set(scores['census_tract_number'])
    tract_set = gdf_tract_set.intersection(
        scores_tract_set,
        set(tracts.values)
        )
    geodata = gdf.loc[gdf['census_tract'].isin(tract_set)]
    data = scores.loc[scores['census_tract_number'].isin(tract_set)]
    map = map_plot(geodata, data)
    return st.markdown(map._repr_html_(), unsafe_allow_html=True)

main()

st.button("")
st.text(" ")
st.markdown('''<p style='text-align: left; color: black; font-size: 28px'>\
    Project insights</p>''',
    unsafe_allow_html=True)
st.markdown(
    '''<p style='text-align: left; color: black; font-size: 20px'><b>Top five \
    important features</b> for identifying the opportunities:</p>''',
    unsafe_allow_html=True)
st.markdown(
    '''<p style='text-align: left; color: black; font-size: 19px'><b>A</b>, \
    loan status (first-time purchase, refinance, or home renavation); \
    <b>B</b>, marriage status (single vs married); <b>C</b>, loan purchaser \
    (institution investors); <b>D</b>, minor (age less than 18 years old) \
    population; <b>E</b>, work travel time (average time spent for a daily \
    round-trip transportation).</p>''',
    unsafe_allow_html=True)
st.text("")
st.markdown(
    '''For details of the project, please vist [here]\
    (https://biomchen.github.io/intelliRefinder.html). If you have any \
    questions or comments, please send them to meng.chen03(at)gmail.com.'''
)
