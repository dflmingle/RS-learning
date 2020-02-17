import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

data = pd.read_csv('./user_log_format1.csv', dtype={'time_stamp':'str'})
data1 = pd.read_csv('./user_info_format1.csv')
data2 = pd.read_csv('./train_format1.csv')
submission = pd.read_csv('./test_format1.csv')
data_train = pd.read_csv('./train_format2.csv')

#data = pd.read_csv('./sample_user_log.csv', dtype={'time_stamp':'str'})
#data1 = pd.read_csv('./sample_user_info.csv')
#data2 = pd.read_csv('./train.csv')
#submission = pd.read_csv('./test.csv')




def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



data=reduce_mem_usage(data)
data1=reduce_mem_usage(data1)
data2=reduce_mem_usage(data2)
data_train=reduce_mem_usage(data_train)

data2['origin'] = 'train'
submission['origin'] = 'test'
matrix = pd.concat([data2, submission], ignore_index=True, sort=False)
matrix.drop(['prob'], axis=1, inplace=True)
#data1是user_info
matrix = matrix.merge(data1, on='user_id', how='left')
#data是user_log
data.rename(columns={'seller_id':'merchant_id'}, inplace=True)

print(matrix.shape)
print(data.shape)
print(data.isnull().sum())
print(matrix.isnull().sum())

'''
data['user_id'] = data['user_id'].astype('int32')
data['merchant_id'] = data['merchant_id'].astype('int32')
data['item_id'] = data['item_id'].astype('int32')
data['cat_id'] = data['cat_id'].astype('int32')
data['brand_id'] = data['brand_id'].astype('int32')
matrix['age_range'] = matrix['age_range'].astype('int8')
matrix['gender'] = matrix['gender'].astype('int8')
matrix['label'] = matrix['label'].astype('str')
matrix['user_id'] = matrix['user_id'].astype('int32')
matrix['merchant_id'] = matrix['merchant_id'].astype('int32')
'''
data['brand_id'].fillna(0, inplace=True)
data['time_stamp'] = pd.to_datetime(data['time_stamp'], format='%m%d')
matrix['age_range'].fillna(0, inplace=True)
matrix['gender'].fillna(2, inplace=True)

del data1, data2
gc.collect()

#特征处理
groups = data.groupby(['user_id'])
temp = groups.size().reset_index().rename(columns={0:'u1'})
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['item_id'].agg([('u2', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['cat_id'].agg([('u3', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['merchant_id'].agg([('u4', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['brand_id'].agg([('u5', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
#temp = groups['time_stamp'].agg([('F_time', 'min'), ('L_time', 'max')]).reset_index()
#temp['u6'] = (temp['L_time'] - temp['F_time']).dt.days
#matrix = matrix.merge(temp[['user_id', 'u6']], on='user_id', how='left')


temp = groups['time_stamp'].agg([('ut', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')




temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'u7', 1:'u8', 2:'u9', 3:'u10'})
matrix = matrix.merge(temp, on='user_id', how='left')



groups = data.groupby(['merchant_id'])
temp = groups.size().reset_index().rename(columns={0:'m1'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
temp = groups['user_id', 'item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(columns={
    'user_id':'m2',
    'item_id':'m3', 
    'cat_id':'m4', 
    'brand_id':'m5'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'m6', 1:'m7', 2:'m8', 3:'m9'})
matrix = matrix.merge(temp, on='merchant_id', how='left')

temp = data_train[data_train['label']==-1].groupby(['merchant_id']).size().reset_index().rename(columns={0:'m10'})
matrix = matrix.merge(temp, on='merchant_id', how='left')

groups = data.groupby(['user_id', 'merchant_id'])
temp = groups.size().reset_index().rename(columns={0:'um1'})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(columns={
    'item_id':'um2',
    'cat_id':'um3',
    'brand_id':'um4'
})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={
    0:'um5',
    1:'um6',
    2:'um7',
    3:'um8'
})
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')
temp = groups['time_stamp'].agg([('frist', 'min'), ('last', 'max')]).reset_index()
temp['um9'] = (temp['last'] - temp['frist']).dt.days
temp.drop(['frist', 'last'], axis=1, inplace=True)
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')


temp = groups['time_stamp'].agg([('umt', 'nunique')]).reset_index()
matrix = matrix.merge(temp, on=['user_id', 'merchant_id'], how='left')

d1=matrix[(matrix.label==1) ]
#print(d1.head())


groups = d1.groupby(['merchant_id'])
temp = groups.size().reset_index().rename(columns={0:'rem1'})
matrix = matrix.merge(temp, on='merchant_id', how='left')


print(matrix.columns.values)

matrix['r1'] = matrix['u9']/matrix['u7'] #用户购买点击比
matrix['r2'] = matrix['m8']/matrix['m6'] #商家购买点击比
matrix['r3'] = matrix['um7']/matrix['um5'] #不同用户不同商家购买点击比
#matrix['r4'] = matrix['u1']/matrix['ut']








matrix.fillna(0, inplace=True)

temp = pd.get_dummies(matrix['age_range'], prefix='age')
matrix = pd.concat([matrix, temp], axis=1)
temp = pd.get_dummies(matrix['gender'], prefix='g')
matrix = pd.concat([matrix, temp], axis=1)
matrix.drop(['age_range', 'gender'], axis=1, inplace=True)

#train、test-setdata
train_data = matrix[matrix['origin'] == 'train'].drop(['origin','user_id','merchant_id'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['label', 'origin','user_id','merchant_id'], axis=1)
train_X, train_y = train_data.drop(['label'], axis=1), train_data['label']

del temp, matrix
gc.collect()

#导入分析库
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn import metrics
'''

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=.3)

model = xgb.XGBClassifier(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42
)



model = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=15, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=1500, objective='binary',metric= 'F1',
    subsample=0.95, colsample_bytree=0.95, subsample_freq=1,
    learning_rate=0.05, random_state=2017
    )

model.fit(
    X_train, 
    y_train,
    eval_metric='auc',
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=10
)
'''

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models
        
    def fit_predict(self, X, y, T):
        increase = True
        print(X.shape)
        if increase:
            pos = pd.Series(y == 1)
            y = pd.Series(y)
            X = pd.concat([X, X.loc[pos]], axis = 0)
            y = pd.concat([y, y.loc[pos]], axis = 0)
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X = X.iloc[idx]
            y = y.iloc[idx]
        print(X.shape)
        print(T.shape)
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        
        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle = True, random_state=17).split(X, y))
        
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        
        
        
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], self.n_splits))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                

                print("fit %s fold %d " %(str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:, 1]
                
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]
            S_test[:, i] = S_test_i.mean(axis=1)
            
        result = cross_val_score(self.stacker, S_train, y, cv=3)
        print("Stacker score : %.5f "%(result.mean()))
       
        self.stacker.fit(S_train, y)

        ypred=self.stacker.predict_proba(S_train)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y, ypred, pos_label=1)
        print( metrics.auc(fpr, tpr))

        X_train, X_valid, y_train, y_valid = train_test_split(S_train, y, test_size=.3)
        model = xgb.XGBClassifier(
            max_depth=8,
            n_estimators=1000,
            min_child_weight=300, 
            colsample_bytree=0.8, 
            subsample=0.8, 
            eta=0.3,    
            seed=42
        )
        model.fit(
            X_train, 
            y_train,
            eval_metric='auc',
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=True,
            early_stopping_rounds=10
        )


        res = self.stacker.predict_proba(S_test)[:, 1]
        return res

lgb_params = {
    'learning_rate' : 0.02,
    'n_estimators' : 650,
    'max_bin' : 10,
    'subsample' : 0.8,
    'subsample_freq' : 10,
    'colsample_bytree' : 0.8,
    'min_child_samples' : 500,
    'seed' : 99
}

lgb_params2 = {
    'n_estimators' : 1090,
    'learning_rate' : 0.02,
    'colsample_bytree' : 0.3,
    'subsample' : 0.7,
    'subsample_freq' : 2,
    'num_leaves' : 16,
    'seed' : 99
}

lgb_params3 = {
    'n_estimators' : 110,
    'max_depth' : 4,
    'learning_rate' : 0.02,
    'seed' : 99
}

lgb_model =lgb. LGBMClassifier(**lgb_params)
lgb_model2 = lgb.LGBMClassifier(**lgb_params2)
lgb_model3 = lgb.LGBMClassifier(**lgb_params3)
xgmodel = xgb.XGBClassifier(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42
)


lgb_model4 = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=15, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=1500, objective='binary',metric= 'F1',
    subsample=0.95, colsample_bytree=0.95, subsample_freq=1,
    learning_rate=0.05, random_state=2017
    )





log_model = LogisticRegression()
stack = Ensemble(n_splits = 3, stacker = log_model, base_models = (lgb_model, lgb_model2,lgb_model3))
y_pred = stack.fit_predict(train_X, train_y, test_data)

prob=y_pred
submission['prob'] = pd.Series(prob)
#prob = model.predict_proba(test_data)
#submission['prob'] = pd.Series(prob[:,1])
submission.drop(['origin'], axis=1, inplace=True)
submission.to_csv('submission.csv', index=False)
