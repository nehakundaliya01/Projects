#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv(r'/Users/nehakundaliya/Downloads/archive (3).zip')
df.head()


# In[6]:


df.shape


# In[7]:


df1 = df[['area_type','location','size','total_sqft','bath','price']]
df1.head()


# In[8]:


df1.isnull().sum()


# As we have lot of data and missing values are less, so we can remove them

# In[9]:


df2 = df1.dropna()


# In[10]:


df2.groupby('area_type')['area_type'].count()


# In[11]:


df2['location'].unique()


# In[12]:


df2['size'].unique()


# In[14]:


df3 = df2.copy()
df3['BHK'] = df2['size'].apply(lambda x: int(x.split()[0]))
df3.head(3)


# In[15]:


df3.drop('size',axis='columns',inplace=True)


# In[16]:


df3.head()


# In[17]:


df3['total_sqft'].unique()


# In[18]:


def is_float(x):
    try:
        float(x)
    except ValueError:
        print(x)


# In[19]:


df3['total_sqft'].apply(is_float)


# In[20]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   


# In[21]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4.head()


# In[22]:


df4['total_sqft'].unique()


# In[23]:


df4.shape


# In[24]:


df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft']
df4.head()


# In[25]:


df5 = df4[df4['total_sqft']/df4['BHK']>=300]


# In[26]:


df5.shape


# In[27]:


df5['price_per_sqft'].describe()


# In[28]:


# You can replace the threshold value as needed
threshold = 10

# Get the location counts and sort them in descending order
location_counts = df5.groupby('location')['location'].count().sort_values(ascending=False).reset_index(name='count')

# Filter locations with count less than the threshold
locations_to_replace = location_counts[location_counts['count'] < threshold]['location'].tolist()

# Replace those locations with 'Other'
df5.loc[df5['location'].isin(locations_to_replace), 'location'] = 'Other'

df5.head()


# In[29]:


df5.shape


# In[30]:


def loc(df):
    df1 = pd.DataFrame()
    for a, b in df.groupby('area_type'):
        for c,d in b.groupby('location'):
            mean = np.mean(d.price_per_sqft)
            std = np.std(d.price_per_sqft)
            ab = d[(d.price_per_sqft <= mean + std) & (d.price_per_sqft >= mean - std)]
            df1 = pd.concat([df1, ab])
    return df1


# In[31]:


loc(df5)


# In[32]:


df6 = loc(df5)



# In[33]:


df6.shape


# In[34]:


df7 = df6[df6['bath']<=df6['BHK']+2]


# In[35]:


df7.shape


# In[36]:


def plot_scatter_chart(df1, location):
        for dc,df in df1.groupby('location'):
            if dc == location:
                bhk2 = df[df.BHK == 2]
                bhk3 = df[df.BHK == 3]
                plt.rcParams['figure.figsize'] = (15, 10)
                plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
                plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
                plt.xlabel("Total Square Feet Area")
                plt.ylabel("Price (Lakh Indian Rupees)")
                plt.title(f"Scatter Plot for {location}")
                plt.legend()
                plt.show()


# In[37]:


plot_scatter_chart(df7,'Banashankari Stage II')


# In[38]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)


# In[39]:


df8.shape


# In[40]:


dummy = pd.get_dummies(df8['area_type'])
dummy


# In[41]:


dummy.columns


# In[42]:


dummy.drop('Super built-up  Area',axis='columns',inplace=True)


# In[43]:


df9 = pd.concat([df8,dummy],axis='columns')


# In[44]:


df9.head()


# In[45]:


dummy2= pd.get_dummies(df9['location'])
dummy2.head(2)


# In[46]:


dummy2.drop('Other',axis='columns',inplace=True)


# In[47]:


df10=pd.concat([df9,dummy2],axis='columns')


# In[48]:


df10.head()


# In[49]:


df11 = df10.drop(['area_type','location','price_per_sqft'],axis='columns')


# In[50]:


df11.shape


# In[51]:


df11.head(3)


# # MODELLING

# In[52]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[53]:


X = df11.drop('price',axis='columns')
Y = df11['price']


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=3)


# In[55]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[56]:


model.score(X_test,y_test)


# In[57]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, Y, cv=cv)


# In[58]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


# In[59]:


def model_selection_gridsearch(x,y):
    algo = {
        'linear':{
            'model':LinearRegression(),
            'params':{
                'fit_intercept': [True, False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
            'alpha': [1,2],
            'selection': ['random', 'cyclic']
        }
        
    },'decision_tree':{
        'model':DecisionTreeRegressor(),
        'params':{
            'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
        }
    }}
    score=[]
    cv =ShuffleSplit(n_splits=5, test_size=0.2, random_state=3)
    for a,b in algo.items():
        ab = GridSearchCV(b['model'],b['params'],cv=cv)
        ab.fit(x,y)
        score.append({
            'model': a,
            'best_score': ab.best_score_,
            'best_params': ab.best_params_
        })
    return pd.DataFrame(score,columns=['model','best_score','best_params'])


# In[60]:


model_selection_gridsearch(X,Y)


# In[62]:


def predict_price(area, location, bhk,bath,sqft):
    # Create an array to represent the input features
    x = np.zeros(len(X.columns))
    
    # Set the values for 'sqft', 'bath', and 'bhk' (assuming 'sqft', 'bath', and 'bhk' are columns in your dataset)
    x[X.columns.get_loc('bath')] = bath
    x[X.columns.get_loc('bhk')] = BHK
    x[X.columns.get_loc('total_sqft')] = total_sqft
    
    # Check if the area is 'super' and set 'a' and 'b' columns accordingly
    if area == 'super':
        x[X.columns.get_loc('a')] = 0
        x[X.columns.get_loc('b')] = 0
    else:
        # Assuming 'a' and 'b' columns are present in your dataset
        x[X.columns.get_loc('a')] = 1  # Set 'a' column to 1 for the provided location
    
    # Use the trained model to predict the price
    predicted_price = lr_clf.predict([x])[0]
    
    return predicted_price


# In[101]:


X.reset_index(drop=True, inplace=True)
X


# In[134]:


def predict_price(area,location,sqft,bath,bhk):
    loc_index=np.where(X.columns==location)[0][0]
    area_index=np.where(X.columns==area)[0][0]
    x = np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    x[loc_index]=1
    x[area_index]=1
    return model.predict([x])[0]


# In[135]:


predict_price('Plot  Area',' Devarachikkanahalli',200,2,2)


# In[136]:


import pickle
with open('linear_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


# In[137]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




