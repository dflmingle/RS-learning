import pandas as pd
import numpy as np

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

train=pd.read_csv('train_ver2.csv')
train=reduce_mem_usage(train)
test=pd.read_csv('test_ver2.csv')
test=reduce_mem_usage(test)

train.drop(['fecha_alta','ult_fec_cli_1t', 'tipodom', 'cod_prov', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1'],axis=1,inplace=True)
test.drop(['fecha_alta','ult_fec_cli_1t', 'tipodom', 'cod_prov'],axis=1,inplace=True)

products=["ind_cco_fin_ult1" , "ind_cder_fin_ult1" ,"ind_cno_fin_ult1"  ,"ind_ctju_fin_ult1", "ind_ctma_fin_ult1"
 ,"ind_ctop_fin_ult1", "ind_ctpp_fin_ult1" ,"ind_dela_fin_ult1" ,"ind_ecue_fin_ult1" ,"ind_fond_fin_ult1",
"ind_hip_fin_ult1" , "ind_plan_fin_ult1" ,"ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1",
"ind_valo_fin_ult1" ,"ind_viv_fin_ult1" , "ind_nomina_ult1"  , "ind_nom_pens_ult1" ,"ind_recibo_ult1" ]
print(len(products))

last_products=[]
for i in products:
    new=i+'_last'
    last_products.append(new)
print(last_products)

last_useful_columns=["ncodpers","tiprel_1mes","ind_actividad_cliente","ind_cco_fin_ult1" , "ind_cder_fin_ult1" ,"ind_cno_fin_ult1"  ,"ind_ctju_fin_ult1", "ind_ctma_fin_ult1"
 ,"ind_ctop_fin_ult1", "ind_ctpp_fin_ult1" ,"ind_dela_fin_ult1" ,"ind_ecue_fin_ult1" ,"ind_fond_fin_ult1",
"ind_hip_fin_ult1" , "ind_plan_fin_ult1" ,"ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1",
"ind_valo_fin_ult1" ,"ind_viv_fin_ult1" , "ind_nomina_ult1"  , "ind_nom_pens_ult1" ,"ind_recibo_ult1"]
print(len(last_useful_columns))

date=train['fecha_dato'].unique().tolist()
date.append('2016-06-28')
print(date)

for i in [5,11,12,13,14,15,16,17]:
    if date[i] != "2016-06-28":
        this_month=train[train['fecha_dato']==date[i]]
        last_month=train[train['fecha_dato']==date[i-1]]
        this_month=pd.merge(this_month,last_month[last_useful_columns],on="ncodpers",how='inner',suffixes=("","_last"))
        this_month=this_month.sort_values(by="ncodpers")
        this_month.to_csv('train2_{}.csv'.format(date[i]),index=False)
    else:
        last_month=train[train['fecha_dato']==date[i-1]]
        this_month=pd.merge(test,last_month[last_useful_columns],on="ncodpers",how='left',suffixes=("","_last"))
        for j in range(len(products)):
            this_month.rename(columns={products[j]:last_products[j]}, inplace=True)
        this_month.to_csv('test2_{}.csv'.format(date[i]),index=False)

for i in [5,11,12,13,14,15,16,17]:
    if date[i] != "2016-06-28":
        this_month=train[train['fecha_dato']==date[i]]['ncodpers']
        last_month=train[train['fecha_dato']==date[i-1]]['ncodpers']
        this_month=pd.merge(this_month,last_month,on="ncodpers",how='inner')
        this_month.sort_values(by="ncodpers" )
    else:
        this_month=test['ncodpers']
        this_month.sort_values(by="ncodpers")
    temp=pd.DataFrame()
    for j in range(i):
        temp=temp.append(train[train['fecha_dato']==date[j]])
    temp=temp.sort_values(by=["ncodpers","fecha_dato"])
    ncodpers_list=temp['ncodpers'].unique().tolist()
    count=pd.DataFrame()
    count=count.append(ncodpers_list)
    count.rename(columns={0:'ncodpers'}, inplace=True)
    for product in products:
        print(product)
        temp1=temp[["ncodpers","fecha_dato",product]]
        temp1=temp1.sort_values(by=["ncodpers","fecha_dato"])
        count_product=pd.DataFrame(columns=(product+'_00',product+'_01',product+'_10',product+'_11',product+'_0len'))
        for m in ncodpers_list:
                count_00=0
                count_01=0
                count_10=0
                count_11=0
                count_0len=0
                one_ncodper=temp1[temp1['ncodpers']==m]
                one_ncodper['dif']=one_ncodper[product].diff()
                one_ncodper['rol']=one_ncodper[product].rolling(2).sum()
                for v in one_ncodper['dif']:
                    if v==1:
                        count_01=count_01+1
                    elif v==-1:
                        count_10=count_10+1
                for u in one_ncodper['rol']:
                    if u==0:
                         count_00=count_00+1
                    elif u==2:
                         count_11=count_11+1
                for w in range(len(one_ncodper[product].tolist())):
                    if one_ncodper[product].tolist()[-1-w]==0:
                        count_0len=count_0len+1
                    else:
                        break
                count_product.loc[len(count_product)] = [count_00,count_01,count_10,count_11,count_0len]
        count=pd.concat([count,count_product[[product+'_00',product+'_01',product+'_10',product+'_11',product+'_0len']]],axis=1)
    count=pd.merge(this_month,count,on='ncodpers',how='left')
    count.drop('ncodpers',axis=1,inplace=True)
    count.to_csv('count2_{}.csv'.format(date[i]),index=False)
