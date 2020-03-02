import numpy as np
import pandas as pd
import time
import sqlite3
from pathlib import Path
import geopandas as gpd
import folium
from geopy.geocoders import Nominatim

def connect_sql(db):
    '''connect to the SQL database'''
    conn = sqlite3.connect(db)
    return conn

def db2sql(db, path, dtype):
    '''dumps the csv into the SQL database'''
    start = time.time()
    conn = connect_sql(db)
    p = Path(path)
    files = list(p.glob('*.{}'.format(dtype)))
    for file in files:
        df = pd.read_csv(file, dtype=object)
        name = str(file).split('/')[-1].split('.')[0]
        print('Creating {} table......'.format(name))
        df.to_sql(name, conn, if_exists='replace')
    duration = time.time() - start
    print('It takes {} to dump the csv file to the SQL database'
          .format(duration))
    conn.commit()
    print('All tables have been created!')

def get_all_transcations(db, columns, years, state_name):
    dat1 = pd.DataFrame()
    conn = connect_sql(db)
    for year in years:
        query = 'SELECT {0} from hmda_{1} WHERE(state_name = "{2}")'
        query = query.format(columns, year, state_name)
        dat2 = pd.read_sql_query(query, conn)
        dat1 = pd.concat([dat1, dat2], axis=0)
    return dat1

def remove_nulls(data, threshold):
    percentages = []
    idxes = []
    # get percentage of the missing value by columns
    for name in data.columns:
        df = data[name]
        percentage = sum(df.isnull())/data.shape[0]
        percentages.append(percentage)

    # remove the columns that have missing value about the threshold
    per_series = pd.Series(percentages)
    for idx, per in per_series.items():
        if per <= threshold:
            idxes.append(idx)
        else:
            continue
    data_new = data.iloc[:, idxes]
    return data_new

def get_feature_codes(df, pairs):
    feat_codes = {}
    for pair in pairs:
        item_codes = {}
        for item, code in zip(df[pair[0]], df[pair[1]]):
            item_codes.update({item:code})
        feat_codes.update({pair[0]:item_codes})
    return feat_codes

def hist_plots(df, i, j):
    column_list = [df.columns[x:x+j] for x in range(0, len(df.columns), j)]
    fig, axes = plt.subplots(i, j, figsize=(16, 21),
                             sharey=True,
                             tight_layout=True)
    for n in range(i):
        for m in range(j):
            column = column_list[n][m]
            df1 = df[column].dropna()
            bins = len(df1.unique())
            axes[n, m].hist(df1, bins=20, color='orange')
            axes[n, m].set_title(column)
            axes[n, m].set_xlabel('')
            axes[n, m].set_ylabel('')

def score2opportunity(df, threshold):
    opportunities = []
    scores = df['Refinance_score']
    for score in scores:
        if score >= threshold:
            opportunities.append(1)
        else:
            opportunities.append(0)
    return opportunities

def feature_pivot(df, features, f_type):
    df_new = pd.DataFrame()
    for f in features:
        if f_type == 'cat':
            func = lambda x: len(x)
            df1 = df[['as_of_year', 'census_tract_number']+[f]]
            df2 = pd.pivot_table(df1,
                                 values='as_of_year',
                                 index=['census_tract_number'],
                                 columns=f,
                                 aggfunc=func, fill_value=0)
            df2.columns = [f+'_'+str(i) for i in df2.columns]

        elif f_type == 'con':
            df1 = df[['census_tract_number']+[f]]
            df2 = df1.groupby(['census_tract_number']).mean()

        df_new = pd.concat([df_new, df2], axis=1)
    return df_new

def get_counts(df, name, col1, col2):
    df1 = df[df[col1] == name]
    df1 = df1[col2].value_counts()
    df2 = df1.reset_index()
    df2.columns = [col2, 'counts']
    return df2

def calculate_score(df_loan_purpose):
    old_columns = df_loan_purpose.columns
    scores = []
    for _, row in df_loan_purpose.iterrows():
        score = row[2]/sum(row)
        scores.append(score)
    df_loan_purpose['Refinance_score'] = scores
    df_loan_purpose.reset_index(inplace=True)
    df_loan_purpose.drop(columns=old_columns, inplace=True)
    return df_loan_purpose

def zip2tract(loc, state_code):
    z2t = {}
    zt = []
    df = pd.read_excel(loc)
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

def get_tract(path='../data/t2z/zip_tract_122017.xlsx'):
    zipcode = input()
    df = pd.read_excel(path)
    df = df[['zip', 'tract']]
    print(df)
    zip_df = df[df['zip'] == int(zipcode)]
    return zip_df['tract']

def get_refinance_score(file):
    df = pd.read_csv(file)
    df.rename(columns={'Unnamed: 0': 'census_tract_number'}, inplace=True)
    df = df[['census_tract_number', 'Refinance_score']]
    return df

def get_geodata(shp):
    gdf = gpd.read_file(shp).to_crs({'init':'epsg:4326'})
    gdf['TRACTCE'] = gdf['TRACTCE'].astype(float)/100
    gdf.rename(columns={'TRACTCE': 'census_tract'}, inplace=True)
    gdf['census_tract'] = gdf['census_tract'].astype(float)
    return gdf

def map_plot(location, geo_data, data):
    plot = folium.Map(location=location,
                      zoom_start=9,
                      width='50%',
                      length='50%')
    plot.choropleth(geo_data=geo_data,
                    name='choropleth',
                    data=data,
                    columns=['census_tract_number', 'Refinance_score'],
                    key_on='feature.properties.census_tract',
                    fill_color='YlOrRd',
                    na_fill_color='white',
                    na_fill_opacity=0.2,
                    fill_opacity=0.7,
                    line_weight=0.6,
                    line_opacity=0.2)
    return plot
