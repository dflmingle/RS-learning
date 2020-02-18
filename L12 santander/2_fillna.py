import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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
train_20150628=pd.read_csv('train2_2015-06-28.csv')
train_20150628=reduce_mem_usage(train_20150628)
train_20151228=pd.read_csv('train2_2015-12-28.csv')
train_20151228=reduce_mem_usage(train_20151228)
train_20160428=pd.read_csv('train2_2016-04-28.csv')
train_20160428=reduce_mem_usage(train_20160428)
train_20160528=pd.read_csv('train2_2016-05-28.csv')
train_20160528=reduce_mem_usage(train_20160528)
test_20160628=pd.read_csv('test2_2016-06-28.csv')
test_20160628=reduce_mem_usage(test_20160628)

train_20150628['ind_empleado'].fillna('N', inplace=True)
train_20151228['ind_empleado'].fillna('N',inplace=True)
train_20160428['ind_empleado'].fillna('N', inplace=True)
train_20160528['ind_empleado'].fillna('N', inplace=True)
test_20160628['ind_empleado'].fillna('N', inplace=True)

train_20150628['pais_residencia'].fillna('ES', inplace=True)
train_20151228['pais_residencia'].fillna('ES', inplace=True)
train_20160428['pais_residencia'].fillna('ES', inplace=True)
train_20160528['pais_residencia'].fillna('ES', inplace=True)
test_20160628['pais_residencia'].fillna('ES', inplace=True)

train_20150628['sexo'].fillna('V', inplace=True)
train_20151228['sexo'].fillna('V', inplace=True)
train_20160428['sexo'].fillna('V', inplace=True)
train_20160528['sexo'].fillna('V', inplace=True)
test_20160628['sexo'].fillna('V', inplace=True)

train_20150628['ind_nuevo'].fillna(0, inplace=True)
train_20151228['ind_nuevo'].fillna(0, inplace=True)
train_20160428['ind_nuevo'].fillna(0, inplace=True)
train_20160528['ind_nuevo'].fillna(0, inplace=True)
test_20160628['ind_nuevo'].fillna(0, inplace=True)

train_20150628['indrel'].fillna(1, inplace=True)
train_20151228['indrel'].fillna(1, inplace=True)
train_20160428['indrel'].fillna(1, inplace=True)
train_20160528['indrel'].fillna(1, inplace=True)
test_20160628['indrel'].fillna(1, inplace=True)

train_20150628['indrel_1mes'].fillna(1, inplace=True)
train_20151228['indrel_1mes'].fillna(1, inplace=True)
train_20160428['indrel_1mes'].fillna(1, inplace=True)
train_20160528['indrel_1mes'].fillna(1, inplace=True)
test_20160628['indrel_1mes'].fillna(1, inplace=True)   #考虑drop

train_20150628['tiprel_1mes'].fillna('I', inplace=True)
train_20151228['tiprel_1mes'].fillna('I', inplace=True)
train_20160428['tiprel_1mes'].fillna('I', inplace=True)
train_20160528['tiprel_1mes'].fillna('I', inplace=True)
test_20160628['tiprel_1mes'].fillna('I', inplace=True)

train_20150628['indresi'].fillna('S', inplace=True)
train_20151228['indresi'].fillna('S', inplace=True)
train_20160428['indresi'].fillna('S', inplace=True)
train_20160528['indresi'].fillna('S', inplace=True)
test_20160628['indresi'].fillna('S', inplace=True)

train_20150628['indext'].fillna('N', inplace=True)
train_20151228['indext'].fillna('N', inplace=True)
train_20160428['indext'].fillna('N', inplace=True)
train_20160528['indext'].fillna('N', inplace=True)
test_20160628['indext'].fillna('N', inplace=True)

train_20150628['conyuemp'].fillna('N', inplace=True)
train_20151228['conyuemp'].fillna('N', inplace=True)
train_20160428['conyuemp'].fillna('N', inplace=True)
train_20160528['conyuemp'].fillna('N', inplace=True)
test_20160628['conyuemp'].fillna('N', inplace=True)  #考虑drop

train_20150628['canal_entrada'].fillna('KHE', inplace=True)
train_20151228['canal_entrada'].fillna('KHE', inplace=True)
train_20160428['canal_entrada'].fillna('KHE', inplace=True)
train_20160528['canal_entrada'].fillna('KHE', inplace=True)
test_20160628['canal_entrada'].fillna('KHE', inplace=True)

train_20150628['indfall'].fillna('N', inplace=True)
train_20151228['indfall'].fillna('N', inplace=True)
train_20160428['indfall'].fillna('N', inplace=True)
train_20160528['indfall'].fillna('N', inplace=True)
test_20160628['indfall'].fillna('N', inplace=True)

train_20150628['nomprov'].fillna('MADRID', inplace=True)
train_20151228['nomprov'].fillna('MADRID', inplace=True)
train_20160428['nomprov'].fillna('MADRID', inplace=True)
train_20160528['nomprov'].fillna('MADRID', inplace=True)
test_20160628['nomprov'].fillna('MADRID', inplace=True)

train_20150628['ind_actividad_cliente'].fillna(0, inplace=True)
train_20151228['ind_actividad_cliente'].fillna(0, inplace=True)
train_20160428['ind_actividad_cliente'].fillna(0, inplace=True)
train_20160528['ind_actividad_cliente'].fillna(0, inplace=True)
test_20160628['ind_actividad_cliente'].fillna(0, inplace=True)

train_20150628['ind_actividad_cliente_last'].fillna(0, inplace=True)
train_20151228['ind_actividad_cliente_last'].fillna(0, inplace=True)
train_20160428['ind_actividad_cliente_last'].fillna(0, inplace=True)
train_20160528['ind_actividad_cliente_last'].fillna(0, inplace=True)
test_20160628['ind_actividad_cliente_last'].fillna(0, inplace=True)

train_20150628['segmento'].fillna('02 - PARTICULARES', inplace=True)
train_20151228['segmento'].fillna('02 - PARTICULARES', inplace=True)
train_20160428['segmento'].fillna('02 - PARTICULARES', inplace=True)
train_20160528['segmento'].fillna('02 - PARTICULARES', inplace=True)
test_20160628['segmento'].fillna('02 - PARTICULARES', inplace=True)

train_20150628['age'].fillna(23, inplace=True)
train_20151228['age'].fillna(23, inplace=True)
train_20160428['age'].fillna(23, inplace=True)
train_20160528['age'].fillna(23, inplace=True)
test_20160628['age'].fillna(23, inplace=True)

train_20150628['antiguedad'].fillna(0, inplace=True)
train_20151228['antiguedad'].fillna(0, inplace=True)
train_20160428['antiguedad'].fillna(0, inplace=True)
train_20160528['antiguedad'].fillna(0, inplace=True)
test_20160628['antiguedad'].fillna(0, inplace=True)

train_20150628['renta'].fillna(train_20150628['renta'].mean(),inplace=True)
train_20151228['renta'].fillna(train_20151228['renta'].mean(),inplace=True)
train_20160428['renta'].fillna(train_20160428['renta'].mean(),inplace=True)
train_20160528['renta'].fillna(train_20160528['renta'].mean(),inplace=True)
test_20160628['renta'].fillna(test_20160628['renta'].mean(),inplace=True)

train_20150628['tiprel_1mes_last'].fillna('I', inplace=True)
train_20151228['tiprel_1mes_last'].fillna('I', inplace=True)
train_20160428['tiprel_1mes_last'].fillna('I', inplace=True)
train_20160528['tiprel_1mes_last'].fillna('I', inplace=True)
test_20160628['tiprel_1mes_last'].fillna('I', inplace=True)

train_20150628.fillna(0, inplace=True)

train_20150628.to_csv('train1_2015-06-28.csv',index=False)
train_20151228.to_csv('train1_2015-12-28.csv',index=False)
train_20160428.to_csv('train1_2016-04-28.csv',index=False)
train_20160528.to_csv('train1_2016-05-28.csv',index=False)
test_20160628.to_csv('test1_2016-06-28.csv',index=False)







