import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import sqlite3
from pathlib import Path
import geopandas as gpd
import folium
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from itertools import chain
from geopy.geocoders import Nominatim


class ProjectModels(object):
    '''
    Attributes:
    x: features vectors
    y: target

    Functions:
    lgr: logitic regression with l1 regularization
    rgr: ridge regression
    lasso: lasso regression
    rft: random forest
    svm: support vector machine
    cross_validate_results: results of 5-fold cross validation
    visualize_coefs: visualize the coefficients or importance of top features
    PCA_val: evaluate the PCs and apply the PCs to different classifiers
    '''
    def __init__(self, x, y, model):
        self.x = x
        self.y = y
        self.model = model
        self.get_model()
        self.lgr()
        self.rgr()
        self.lasso()
        self.rft()
        self.cross_validate_results()
        self.predict_results()

    def get_model(self):
        models = {'lgr': self.lgr(),
                  'rgr': self.rgr(),
                  'lasso': self.lasso(),
                  'rft': self.rft()}
        return models[self.model]

    # logistic regressino with l1 regularization
    def lgr(self, max_iter=10000, penalty='l1', solver='saga'):
        model = LogisticRegression(
            random_state=50,
            max_iter=max_iter,
            penalty=penalty,
            solver=solver)
        model.fit(self.x, self.y)
        return model

    # ridged regression for l2 regularization
    def rgr(self, alpha=1, solver='auto'):
        model = Ridge(alpha=alpha, solver=solver)
        model.fit(self.x, self.y)
        return model

    # lasso regression
    def lasso(self, alpha=1, max_iter=1000):
        model = Lasso(alpha=alpha, max_iter=max_iter)
        model.fit(self.x, self.y)
        return model

    # random forest
    def rft(self, n_estimators=500, mx_leaf_nodes=16, n_jobs=-1):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_leaf_nodes=mx_leaf_nodes,
            n_jobs=n_jobs)
        model.fit(self.x, self.y)
        return model

    # cross validate for regression models
    def cross_validate_results(self, cv=5):
        # using precision, recall and f1 scores as validated matrices
        # with 5 folds of the data
        scoring = ['precision', 'recall', 'f1']
        model = self.get_model()
        scores = cross_validate(
            model,
            self.x, self.y,
            cv=cv,
            scoring=scoring,
            return_train_score=False)
        # calculate the mean values of each matrices
        # return it to the value of the keys (matrices)
        for key in scores.keys():
            scores[key] = round(np.mean(scores[key]), 3)
        return scores

    # predict the results
    def predict_results(self):
        model = self.get_model()
        results = model.predict(self.x)
        return results

    # visualize the coeffients or importance of features
    def visualize_coefs(self, num):
        model = self.get_model()
        if self.model == 'rft':
            coefs = model.feature_importances_
        else:
            coefs = model.coef_
            coefs = [coef for coef in chain(*coefs)]
        coef_df = pd.DataFrame([self.x.columns, coefs]).T
        coef_df.columns=['feature', 'coefs']
        coef_df['abs'] = abs(coef_df.coefs)
        coef_df.sort_values(by='abs', ascending=False, inplace=True)
        # set up the fig settings
        fig, ax = plt.subplots(figsize=(6, num*0.3))
        # set up the seaborn settings
        sns.set(style='whitegrid')
        sns.barplot(y='feature', x='abs',
                    data=coef_df.iloc[:num],
                    color='b')

    # pca evaluation
    def PCA_eval(self, n):
        pca = PCA(n_components=n)
        x = pca.fit_transform(self.x)
        print(pca.explained_variance_ratio_)
        sns.scatterplot(x[:, 0], x[:, 1], hue=self.y)
        plt.show()
        # PC analysis
        c = pca.components_
        df = pd.DataFrame(
            pca.components_,
            columns=self.x.columns,
            index=['PC1','PC2','PC3']).T
        feat_names = []
        for i in range(n):
            pc = df.iloc[:, i]
            for col, val in zip(pc.index, abs(pc.values)):
                if val >= 0.2:
                    feat_names.append(col)
                else:
                    continue
        return feat_names

    # pca cross validation
    def PCA_cross_validate(self, n, cv=5):
        scoring = ['precision', 'recall', 'f1']
        pca = PCA(n_components=n)
        x = pca.fit_transform(self.x)
        model = self.get_model()
        scores = cross_validate(
            model,
            x, self.y,
            cv=cv,
            scoring=scoring,
            return_train_score=False)
        # calculate the mean values of each matrices
        # return it to the value of the keys (matrices)
        for key in scores.keys():
            scores[key] = round(np.mean(scores[key]), 3)
        return scores

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
            df2 = pd.pivot_table(
                df1,
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
    plot = folium.Map(
        location=location,
        zoom_start=9,
        width='50%',
        length='50%')
    plot.choropleth(
        geo_data=geo_data,
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
