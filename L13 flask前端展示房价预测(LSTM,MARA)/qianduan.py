# coding=gbk
from flask import Flask,url_for,request,render_template
from flask_wtf import FlaskForm
from wtforms import  SubmitField, SelectField,StringField
from wtforms.validators import DataRequired
import pandas as pd 

area1=[(1,'��ƽ��'),(2,'������'),(3,'������'),(4,'������'),(5,'������'),(6,'��ɽ��'),(7,'��̨��'),(8,'������'),(9,'������'),(10,'��ͷ����'),(11,'������'),(12,'ƽ����'),(13,'ʯ��ɽ��'),(14,'˳����'),(15,'ͨ����'),(16,'������'),(17,'������'),(18,'������')]
area2=[(1,'������'),(2,'�ӻ���'),(3,'��خ��'),(4,'������'),(5,'������'),(6,'������'),(7,'������'),(8,'�ܸ���'),(9,'��ɳ��'),(10,'�����'),(11,'Խ����'),(12,'������')]
area3=[(1,'������'),(2,'������'),(3,'�����'),(4,'������'),(5,'�ζ���'),(6,'��ɽ��'),(7,'������'),(8,'¬����'),(9,'������'),(10,'�ֶ�����'),(11,'������'),(12,'������'),(13,'�ɽ���'),(14,'�����'),(15,'������'),(16,'բ����'),(17,'������'),(18,'��ɽ��')]
area4=[(1,'������'),(2,'������'),(3,'������'),(4,'�޺���'),(5,'��ɽ��'),(6,'������')]


BeiJing={}
GuangZhou={}
ShangHai={}
ShenZhen={}
for i,area in area1:
   BeiJing[i]=area 
for i,area in area2:
   GuangZhou[i]=area
for i,area in area3:
   ShangHai[i]=area
for i,area in area4:
   ShenZhen[i]=area





class LoginForm(FlaskForm):
    
    select1= SelectField(choices=area1,coerce=int)
    submit1 = SubmitField('ȷ��')

    select2= SelectField(choices=area2,coerce=int)
    submit2 = SubmitField('ȷ��')
 
    select3= SelectField(choices=area3,coerce=int)
    submit3 = SubmitField('ȷ��')

    select4= SelectField(choices=area4,coerce=int)
    submit4 = SubmitField('ȷ��')

app= Flask(__name__, static_folder="templates")
app.config["SECRET_KEY"] = "12345678"

@app.route("/",methods=['GET','POST'])
def index():
    form = LoginForm()
    returnData={}
    if form.validate_on_submit():
         
         if form.submit1.data:
              print(BeiJing[form.select1.data])
              d1=pd.read_csv("%s,%s.csv"%('����',BeiJing[form.select1.data]),index_col='Unnamed: 0')
              print(d1.head())
              returnData['index']=list(d1.index)
              
              returnData['forecast']=list(d1['forecast'])
              
              returnData['area']='����'+BeiJing[form.select1.data]
              returnData['truth']=list(d1['����(Ԫ/�O)'])
              returnData['truth']= returnData['truth'][0:-3]
         if form.submit2.data:
              print(GuangZhou[form.select2.data])             
              d1=pd.read_csv("%s,%s.csv"%('����',GuangZhou[form.select2.data]),index_col='Unnamed: 0')
              print(d1.head())
              returnData['index']=list(d1.index)
              
              returnData['forecast']=list(d1['forecast'])
              returnData['area']='����'+GuangZhou[form.select2.data]
              returnData['truth']=list(d1['����(Ԫ/�O)'])
              returnData['truth']= returnData['truth'][0:-3]
         if form.submit3.data:
              print(ShangHai[form.select3.data])
              d1=pd.read_csv("%s,%s.csv"%('�Ϻ�',ShangHai[form.select3.data]),index_col='Unnamed: 0')
              
              print(d1.head())
              returnData['index']=list(d1.index)
              returnData['area']='�Ϻ�'+ShangHai[form.select3.data]
              returnData['forecast']=list(d1['forecast'])

              returnData['truth']=list(d1['����(Ԫ/�O)'])
              returnData['truth']= returnData['truth'][0:-3]

         if form.submit4.data:
              print(ShenZhen[form.select4.data])
              d1=pd.read_csv("%s,%s.csv"%('����',ShenZhen[form.select4.data]),index_col='Unnamed: 0')
              print(d1.head())
              returnData['index']=list(d1.index)

              returnData['forecast']=list(d1['forecast'])
              returnData['area']='����'+ShenZhen[form.select4.data]
              returnData['truth']=list(d1['����(Ԫ/�O)'])
              returnData['truth']= returnData['truth'][0:-3]
    return render_template('index.html',form=form,returnData=returnData)



if __name__ == "__main__":
    app.run()
