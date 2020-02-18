# coding=gbk
from flask import Flask,url_for,request,render_template
from flask_wtf import FlaskForm
from wtforms import  SubmitField, SelectField,StringField
from wtforms.validators import DataRequired
import pandas as pd 

area1=[(1,'昌平区'),(2,'朝阳区'),(3,'崇文区'),(4,'大兴区'),(5,'东城区'),(6,'房山区'),(7,'丰台区'),(8,'海淀区'),(9,'怀柔区'),(10,'门头沟区'),(11,'密云区'),(12,'平谷区'),(13,'石景山区'),(14,'顺义区'),(15,'通州区'),(16,'西城区'),(17,'宣武区'),(18,'延庆区')]
area2=[(1,'白云区'),(2,'从化区'),(3,'番禺区'),(4,'海珠区'),(5,'花都区'),(6,'黄埔区'),(7,'荔湾区'),(8,'萝岗区'),(9,'南沙区'),(10,'天河区'),(11,'越秀区'),(12,'增城区')]
area3=[(1,'崇明区'),(2,'奉贤区'),(3,'虹口区'),(4,'黄浦区'),(5,'嘉定区'),(6,'金山区'),(7,'静安区'),(8,'卢湾区'),(9,'闵行区'),(10,'浦东新区'),(11,'普陀区'),(12,'青浦区'),(13,'松江区'),(14,'徐汇区'),(15,'杨浦区'),(16,'闸北区'),(17,'长宁区'),(18,'宝山区')]
area4=[(1,'宝安区'),(2,'福田区'),(3,'龙岗区'),(4,'罗湖区'),(5,'南山区'),(6,'盐田区')]


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
    submit1 = SubmitField('确定')

    select2= SelectField(choices=area2,coerce=int)
    submit2 = SubmitField('确定')
 
    select3= SelectField(choices=area3,coerce=int)
    submit3 = SubmitField('确定')

    select4= SelectField(choices=area4,coerce=int)
    submit4 = SubmitField('确定')

app= Flask(__name__, static_folder="templates")
app.config["SECRET_KEY"] = "12345678"

@app.route("/",methods=['GET','POST'])
def index():
    form = LoginForm()
    returnData={}
    if form.validate_on_submit():
         
         if form.submit1.data:
              print(BeiJing[form.select1.data])
              d1=pd.read_csv("%s,%s.csv"%('北京',BeiJing[form.select1.data]),index_col='Unnamed: 0')
              print(d1.head())
              returnData['index']=list(d1.index)
              
              returnData['forecast']=list(d1['forecast'])
              
              returnData['area']='北京'+BeiJing[form.select1.data]
              returnData['truth']=list(d1['供给(元/㎡)'])
              returnData['truth']= returnData['truth'][0:-3]
         if form.submit2.data:
              print(GuangZhou[form.select2.data])             
              d1=pd.read_csv("%s,%s.csv"%('广州',GuangZhou[form.select2.data]),index_col='Unnamed: 0')
              print(d1.head())
              returnData['index']=list(d1.index)
              
              returnData['forecast']=list(d1['forecast'])
              returnData['area']='广州'+GuangZhou[form.select2.data]
              returnData['truth']=list(d1['供给(元/㎡)'])
              returnData['truth']= returnData['truth'][0:-3]
         if form.submit3.data:
              print(ShangHai[form.select3.data])
              d1=pd.read_csv("%s,%s.csv"%('上海',ShangHai[form.select3.data]),index_col='Unnamed: 0')
              
              print(d1.head())
              returnData['index']=list(d1.index)
              returnData['area']='上海'+ShangHai[form.select3.data]
              returnData['forecast']=list(d1['forecast'])

              returnData['truth']=list(d1['供给(元/㎡)'])
              returnData['truth']= returnData['truth'][0:-3]

         if form.submit4.data:
              print(ShenZhen[form.select4.data])
              d1=pd.read_csv("%s,%s.csv"%('深圳',ShenZhen[form.select4.data]),index_col='Unnamed: 0')
              print(d1.head())
              returnData['index']=list(d1.index)

              returnData['forecast']=list(d1['forecast'])
              returnData['area']='深圳'+ShenZhen[form.select4.data]
              returnData['truth']=list(d1['供给(元/㎡)'])
              returnData['truth']= returnData['truth'][0:-3]
    return render_template('index.html',form=form,returnData=returnData)



if __name__ == "__main__":
    app.run()
