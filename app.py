
from refinder import *
import streamlit as st
import folium
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# import pandas as pd
# import psycopg2

# Model saved in pickle
DATA_hmda_acs = 'data/hmda_acs_merged.csv'
DATA_zip_tract = 'data/zip_tract_122017.xlsx'
DATA_shp = 'data/2019/tl_2019_53_tract.shp'
MODEL_path = 'models/finalized_model.sav'

#sscaler = StandardScaler()
transformer = FunctionTransformer(np.log1p, validate=True)
model = pickle.load(open(MODEL_path, 'rb'))

@st.cache(persist=True, suppress_st_warning=True)
def get_tract3(zipcode, file=DATA_zip_tract):
    zipcode = int(zipcode)
    results = zip2tract(file, 53)[zipcode]
    return results

@st.cache(persist=True, suppress_st_warning=True)
def model_predict(zipcode):
    tracts = get_tract3(zipcode)
    df = pd.read_csv(DATA_hmda_acs)
    df.rename(columns={'Unnamed: 0': 'census_tract_number'}, inplace=True)
    idxs = df['census_tract_number'].isin(tracts.values)
    tracts = df[idxs].iloc[:, 0]
    df = df[idxs].iloc[:, 1:-2]
    x = transformer.transform(df)
    results = model.predict(x)
    tracts = np.asarray(tracts)
    results_merged = pd.DataFrame({'tracts': tracts, 'results': results})
    results_merged.rename(columns={'tracts':'census_tract_number',
                                   'results':'Refinance_score'}, inplace=True)
    return results_merged


def main():
    st.title('Mortgage Refinance Opportunity')
    st.write('')
    zipcode = st.text_input(label='Input zip code, for example: 98105', value='98105')
    st.text('STATUS: Gathering mortgage transcation and census informaton...')
    tracts = get_tract3(zipcode)
    st.text('STATUS: Engineering features...')
    finance_scores = model_predict(zipcode)
    st.text('STAUTS: Locating information base on local zip codes...')
    gdf = get_geodata(DATA_shp)
    geodata = gdf.loc[gdf['census_tract'].isin(tracts.values)]
    data = finance_scores.loc[finance_scores['census_tract_number']
                              .isin(tracts.values)]
    st.text('STATUS: Generating maps of mortgage refinance opportunities...')
    map = map_plot([47.6062, -122.3321], geodata , data)

    return st.markdown(map._repr_html_(), unsafe_allow_html=True)
    #return None

if __name__ == '__main__':
    main()
