import pandas as pd
import gc
from itertools import combinations
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

products=["ind_cco_fin_ult1" , "ind_cder_fin_ult1" ,"ind_cno_fin_ult1"  ,"ind_ctju_fin_ult1", "ind_ctma_fin_ult1"
 ,"ind_ctop_fin_ult1", "ind_ctpp_fin_ult1" ,"ind_dela_fin_ult1" ,"ind_ecue_fin_ult1" ,"ind_fond_fin_ult1",
"ind_hip_fin_ult1" , "ind_plan_fin_ult1" ,"ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1",
"ind_valo_fin_ult1" ,"ind_viv_fin_ult1" , "ind_nomina_ult1"  , "ind_nom_pens_ult1" ,"ind_recibo_ult1" ]
print(len(products))

lbe = LabelEncoder()
for product in products:
    print(product)
    if product == "ind_cco_fin_ult1":
        train_date = '2015-12-28'
    elif product == "ind_reca_fin_ult1":
        train_date = '2015-06-28'
    else:
        train_date = '2016-05-28'
    data_1 = pd.read_csv('train1_{}.csv'.format(train_date))
    data_2 = pd.read_csv('count2_{}.csv'.format(train_date))
    data_train = pd.concat([data_1, data_2], axis=1)
    if train_date == '2016-05-28':
        data_1 = pd.read_csv('train1_2016-04-28.csv')
        data_2 = pd.read_csv('count2_2016-04-28.csv')
    else:
        data_1 = pd.read_csv('train1_2016-05-28.csv')
        data_2 = pd.read_csv('count2_2016-05-28.csv')
    data_val = pd.concat([data_1, data_2], axis=1)
    data_1 = pd.read_csv('test1_2016-06-28.csv')
    data_2 = pd.read_csv('count2_2016-06-28.csv')
    data_test = pd.concat([data_1, data_2], axis=1)
    del data_1, data_2
    gc.collect()

    data_train = data_train[data_train[product + '_last'] == 0]
    data_val = data_val[data_val[product + '_last'] == 0]
    data_test = data_test[data_test[product + '_last'] == 0]

    data_train.loc[data_train['age'] >= 60, 'age'] = 3
    data_train.loc[(data_train['age'] < 60) & (data_train['age'] >= 40), 'age'] = 2
    data_train.loc[(data_train['age'] < 40) & (data_train['age'] >= 20), 'age'] = 1
    data_train.loc[data_train['age'] < 20, 'age'] = 0
    data_test.loc[data_test['age'] >= 60, 'age'] = 3
    data_test.loc[(data_test['age'] < 60) & (data_test['age'] >= 40), 'age'] = 2
    data_test.loc[(data_test['age'] < 40) & (data_test['age'] >= 20), 'age'] = 1
    data_test.loc[data_test['age'] < 20, 'age'] = 0
    data_val.loc[data_val['age'] >= 60, 'age'] = 3
    data_val.loc[(data_val['age'] < 60) & (data_val['age'] >= 40), 'age'] = 2
    data_val.loc[(data_val['age'] < 40) & (data_val['age'] >= 20), 'age'] = 1
    data_val.loc[data_val['age'] < 20, 'age'] = 0

    data_train['n_products'] = 0
    data_val['n_products'] = 0
    data_test['n_products'] = 0
    for j in range(20):
        data_train['n_products'] = data_train['n_products'] + data_train.iloc[:, j + 42]
        data_val['n_products'] = data_val['n_products'] + data_val.iloc[:, j + 42]
        data_test['n_products'] = data_test['n_products'] + data_test.iloc[:, j + 22]

    data_train['products_last'] = ""
    data_val['products_last'] = ""
    data_test['products_last'] = ""
    for j in range(20):
        data_train['products_last'] = data_train['products_last'].astype(str) + '_' + data_train.iloc[:, j + 42].astype(
            str)
        data_val['products_last'] = data_val['products_last'].astype(str) + '_' + data_val.iloc[:, j + 42].astype(str)
        data_test['products_last'] = data_test['products_last'].astype(str) + '_' + data_test.iloc[:, j + 22].astype(
            str)

    data_train['count_' + 'products_last'] = data_train.groupby(['products_last'])['products_last'].transform('count')
    data_test['count_' + 'products_last'] = data_test.groupby(['products_last'])['products_last'].transform('count')
    data_val['count_' + 'products_last'] = data_val.groupby(['products_last'])['products_last'].transform('count')

    data_train['products_last'] = lbe.fit_transform(data_train['products_last'].astype(str))
    data_val['products_last'] = lbe.fit_transform(data_val['products_last'].astype(str))
    data_test['products_last'] = lbe.fit_transform(data_test['products_last'].astype(str))

    cat_cols = [product + '_00', product + '_01', product + '_11', product + '_10']
    for col in combinations(cat_cols, 2):
        data_train[str(col[0]) + '_' + str(col[1])] = data_train[col[0]].astype(str) + '_' + data_train[col[1]].astype(
            str)
        data_test[str(col[0]) + '_' + str(col[1])] = data_test[col[0]].astype(str) + '_' + data_test[col[1]].astype(
            str)
        data_val[str(col[0]) + '_' + str(col[1])] = data_val[col[0]].astype(str) + '_' + data_val[col[1]].astype(
            str)

    data_train['ind_actividad_cliente_from_to'] = data_train['ind_actividad_cliente_last'].astype(str) + '_' + \
                                                  data_train['ind_actividad_cliente'].astype(str)
    data_val['ind_actividad_cliente_from_to'] = data_val['ind_actividad_cliente_last'].astype(str) + '_' + data_val[
        'ind_actividad_cliente'].astype(str)
    data_test['ind_actividad_cliente_from_to'] = data_test['ind_actividad_cliente_last'].astype(str) + '_' + data_test[
        'ind_actividad_cliente'].astype(str)
    data_train['tiprel_1mes_from_to'] = data_train['tiprel_1mes_last'].astype(str) + '_' + data_train[
        'tiprel_1mes'].astype(str)
    data_val['tiprel_1mes_from_to'] = data_val['tiprel_1mes_last'].astype(str) + '_' + data_val['tiprel_1mes'].astype(
        str)
    data_test['tiprel_1mes_from_to'] = data_test['tiprel_1mes_last'].astype(str) + '_' + data_test[
        'tiprel_1mes'].astype(str)

    data_train.drop('ind_actividad_cliente_last', axis=1, inplace=True)
    data_train.drop('tiprel_1mes_last', axis=1, inplace=True)
    data_test.drop('ind_actividad_cliente_last', axis=1, inplace=True)
    data_test.drop('tiprel_1mes_last', axis=1, inplace=True)
    data_val.drop('ind_actividad_cliente_last', axis=1, inplace=True)
    data_val.drop('tiprel_1mes_last', axis=1, inplace=True)

    cat_col = [i for i in data_test.select_dtypes(object).columns if i not in ['ncodpers', 'fecha_dato']]
    for i in cat_col:
        data_train[i] = lbe.fit_transform(data_train[i].astype(str))
        data_val[i] = lbe.fit_transform(data_val[i].astype(str))
        data_test[i] = lbe.fit_transform(data_test[i].astype(str))

    exp_var = data_test.columns.tolist()[2:]
    x_train = data_train[exp_var]
    y_train = data_train[product]
    x_val = data_val[exp_var]
    y_val = data_val[product]
    x_test = data_test[exp_var]
    model = CatBoostClassifier(learning_rate=0.05, n_estimators=1000, random_state=2019)
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)])
    data_val[product] = model.predict_proba(x_val)[:, 1]
    new_val = data_val[['ncodpers', product]]
    new_val = new_val.sort_values(by='ncodpers')
    new_val.to_csv("catboost_validation_{}_{}.csv".format(product, train_date), index=False)
    data_test[product] = model.predict_proba(x_test)[:, 1]
    new_test = data_test[['ncodpers', product]]
    new_test = new_test.sort_values(by='ncodpers')
    new_test.to_csv("catboost_submission_{}_{}.csv".format(product, train_date), index=False)
