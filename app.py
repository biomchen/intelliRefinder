#
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import pickle
import numpy as np
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer

# Model saved in pickle
DATA_hmda_acs = 'hmda_acs_merged.csv'
DATA_zip_tract = 'zip_tract_122017.xlsx'
DATA_shp = '2019/tl_2019_53_tract.shp'
DATA_zipcodes = 'zipcodes_king.csv'
MODEL_lr_nh = 'lr_model_nh.sav'
MODEL_lr_hf = 'lr_model_hf.sav'
MODEL_rf_nh = 'rf_model_nh.sav'
MODEL_rf_hf = 'rf_model_hf.sav'

model_dict = {'Logistic Regression': {'Non-human': MODEL_lr_nh,
                                      'Human-first': MODEL_lr_hf},
              'Random Forest': {'Non-human': MODEL_rf_nh,
                                'Human-first': MODEL_rf_hf}
            }

transformer = FunctionTransformer(np.log1p, validate=True)
scaler = MinMaxScaler(feature_range=(0.2, 0.8))

st.title('intelliRefinder')
st.markdown('''Finding refinance opportunities for mortgage lenders using
               machine learning algorithms.
            ''')

zipcodes = pd.read_csv(DATA_zipcodes)['zip']
algorithms = ('Logistic Regression', 'Random Forest')
interventions = ('Non-human', 'Human-first')

st.sidebar.markdown('Data availability: King County, WA')
zipcode = st.sidebar.selectbox('Please select zip code', zipcodes)
algo = st.sidebar.selectbox('Select algorithm', algorithms)
itvn = st.sidebar.selectbox('Intervention',interventions)

@st.cache(persist=True, suppress_st_warning=True)
def zip2tract(state_code=53):
    z2t = {}
    zt = []
    df = pd.read_excel(DATA_zip_tract)
    df.columnns = ['zip', 'tract']
    for z, t in zip(df['zip'], df['tract']):
        if str(t)[:2] == str(state_code):
            zt.append([z, str(t)[-5:]])
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

def get_tract(zipcode):
    zipcode = int(zipcode)
    return zip2tract()[zipcode]

def select_model(algo, itvn):
    return model_dict[algo][itvn]

@st.cache(persist=True, suppress_st_warning=True)
def load_model(algo, itvn):
    model_path = select_model(algo, itvn)
    return pickle.load(open(model_path, 'rb'))

@st.cache(persist=True, suppress_st_warning=True)
def load_data(itvn):
    df = pd.read_csv(DATA_hmda_acs)
    ethical_related = df.columns[41:80]
    if itvn == 'Non-human':
        df.drop(['loan_purpose_1', 'loan_purpose_2', 'loan_purpose_3'],
                axis=1,inplace=True)
    elif itvn == 'Human-first':
        df.drop(ethical_related, axis=1, inplace=True)
    df.rename(columns={'Unnamed: 0': 'census_tract_number'}, inplace=True)
    return df

def predict(tracts, algo, itvn):
    df = load_data(itvn)
    df_x = df.iloc[:, :-2]
    idxs = df_x['census_tract_number'].isin(tracts.values)
    tracts = df_x[idxs].iloc[:, 0]
    df_x = df_x[idxs].iloc[:, 1:]
    scores = df[idxs].iloc[:, -2]
    x = transformer.transform(df_x)
    model = load_model(algo, itvn)
    results = model.predict(x)
    results = results + scores
    results = scaler.fit_transform(np.asarray(results).reshape(-1,1))
    results = [i for i in chain(*results)]
    tracts = np.asarray(tracts)
    results_merged = pd.DataFrame({'tracts': tracts, 'results': results})
    results_merged.rename(columns={'tracts':'census_tract_number',
                                   'results':'Refinance_score'}, inplace=True)
    return results_merged

@st.cache(persist=True, suppress_st_warning=True)
def get_geodata(shp):
    gdf = gpd.read_file(shp).to_crs({'init':'epsg:4326'})
    gdf['TRACTCE'] = gdf['TRACTCE'].astype(float)/100
    gdf.rename(columns={'TRACTCE': 'census_tract'}, inplace=True)
    gdf['census_tract'] = gdf['census_tract'].astype(float)
    return gdf

def map_plot(geo_data, data):
    lat = geo_data['INTPTLAT'].astype(float).mean()
    lon = geo_data['INTPTLON'].astype(float).mean()
    plot = folium.Map([lat, lon],
                      zoom_start=11)
    plot.choropleth(geo_data=geo_data,
                    name='choropleth',
                    data=data,
                    columns=['census_tract_number', 'Refinance_score'],
                    key_on='feature.properties.census_tract',
                    fill_color='YlGnBu',
                    legend_name='Mortgage refinance score',
                    na_fill_color='white',
                    na_fill_opacity=0.2,
                    fill_opacity=0.7,
                    line_weight=0.6,
                    line_opacity=0.2)
    return plot

def main():
    tracts = get_tract(zipcode)
    scores = predict(tracts, algo, itvn)
    gdf = get_geodata(DATA_shp)
    geodata = gdf.loc[gdf['census_tract'].isin(tracts.values)]
    data = scores.loc[scores['census_tract_number'].isin(tracts.values)]
    map = map_plot(geodata , data)

    return st.markdown(map._repr_html_(), unsafe_allow_html=True)
    #return None

main()

st.markdown('''If you have questions about or are interested in the project,
               please feel free to contact me @meng.chen03(at)gmail.com.
            ''')
