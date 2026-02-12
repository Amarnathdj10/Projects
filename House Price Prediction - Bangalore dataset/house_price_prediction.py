import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso

df1 = pd.read_csv(r'D:\Coding journey\Projects\Bangalore Home Prices - Real Estate Price Prediction System\bengaluru_house_prices.csv')
df1.groupby('area_type')['area_type'].agg('count')
df2 = df1.drop(['area_type','availability','society','balcony'],axis=1)
df3 = df2.dropna().copy()
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


def convert_to_sqft(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_to_sqft)

df5 = df4.copy()
df5['price_per_sqft'] = (df5['price']*100000)/df5['total_sqft']

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)

location_stats_less_than_10 = location_stats[location_stats<=10]

df5.location = df5.location.apply(lambda x: 'Other' if x in location_stats_less_than_10 else x)

df6 = df5.copy()
df6 = df5[~(df5.total_sqft/df5.bhk<300)]

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft < (m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)

def remove_bedroom_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')


df8 = df7.copy()
df8 = remove_bedroom_outliers(df7)

df9 = df8.copy()
df9 = df8[df8.bath<df8.bhk+2]
df9 = df9.drop(['price_per_sqft','size'],axis=1)

dummies = pd.get_dummies(df9.location).astype(int)

df10 = pd.concat([df9,dummies.drop('Other',axis=1)],axis=1)
df12 = df10.drop('location',axis=1)

X = df12.drop('price',axis=1)
y = df12.price

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)

cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=42)


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, LinearRegression

def find_best_model(X,y):
    model_params = {
        'Linear Regression' : {
            'model' : LinearRegression(),
            'params' : {}
        },
        'Lasso Regression' : {
            'model': Lasso(),
            'params': {
                'alpha': [0.1,1,10],
                'selection': ['random','cyclic']
            }
        },
        'Decision Tree Regressor': {
            'model' : DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error','friedman_mse'],
                'splitter': ['best','random']
,            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5,random_state=42,test_size=0.2)
    for model_name,mp in model_params.items():
        grid = GridSearchCV(mp['model'],mp['params'],cv=cv,return_train_score=False)
        grid.fit(X,y)
        scores.append({
            'model': model_name,
            'best_score': grid.best_score_,
            'best_params': grid.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model(X,y)

def predict_price(location, sqft, bath, bhk):

    x = pd.DataFrame(columns=X.columns)
    
    x.loc[0] = 0

    x.loc[0, 'total_sqft'] = sqft
    x.loc[0, 'bath'] = bath
    x.loc[0, 'bhk'] = bhk

    if location in X.columns:
        x.loc[0, location] = 1

    return lr.predict(x)[0]


print(predict_price('Indira Nagar',1000,3,3))