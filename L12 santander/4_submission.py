import pandas as pd

submission=pd.read_csv('sample_submission.csv')
fin_submission=pd.read_csv('sample_submission.csv')

products=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1',"ind_cco_fin_ult1" , "ind_cder_fin_ult1" ,"ind_cno_fin_ult1"  ,"ind_ctju_fin_ult1", "ind_ctma_fin_ult1"
 ,"ind_ctop_fin_ult1", "ind_ctpp_fin_ult1" ,"ind_dela_fin_ult1" ,"ind_ecue_fin_ult1" ,"ind_fond_fin_ult1",
"ind_hip_fin_ult1" , "ind_plan_fin_ult1" ,"ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1",
"ind_valo_fin_ult1" ,"ind_viv_fin_ult1" , "ind_nomina_ult1"  , "ind_nom_pens_ult1" ,"ind_recibo_ult1"]
print(len(products))

for product in products:
    if product in ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1']:
        temp=submission
        temp[product]=0.0000000001
        submission=pd.concat([submission,temp[product]],axis=1)
    else:
        if product=='ind_cco_fin_ult1':
                train_date='2015-12-28'
        elif product=='ind_reca_fin_ult1':
            train_date='2015-06-28'
        else:
            train_date='2016-05-28'
        temp=pd.read_csv("catboost_submission_{}_{}.csv".format(product, train_date))
        submission=pd.concat([submission,temp[product]],axis=1)
submission1=submission[products].to_dict(orient='records')
rec_products=pd.DataFrame(columns=['products_0','products_1','products_2','products_3','products_4','products_5','products_6'])
for i in submission1:
    result=sorted(i.items(),key=lambda k:k[1],reverse=True)[:7]
    rec=[]
    for j in result:
        rec.append(j[0])
    rec_products.loc[len(rec_products)] = rec
rec_products['added_products1']=rec_products['products_0'].astype(str)+','+rec_products['products_1']+','+rec_products['products_2']+','+rec_products['products_3']+','+rec_products['products_4']+','+rec_products['products_5']+','+rec_products['products_6']
fin_submission=pd.concat([fin_submission,rec_products['added_products1']],axis=1)
fin_submission.drop('added_products',axis=1,inplace=True)
fin_submission.rename(columns={'added_products1':'added_products'},inplace=True)
fin_submission.to_csv('submission77.csv',index=False)