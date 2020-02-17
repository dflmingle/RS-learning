from matrix import Matrix
from als import ALS
import os
from collections import defaultdict

def load_movie_ratings():

    file_name = "combined_data_1"
    path = os.path.join( "%s.txt" % file_name)
    f = open(path)
    lines = iter(f)
    head=[]
    data=[]
    for line in lines:

       row=[]
       temp=line[:-1].split(",")
       if len(temp)>1:
          temp=[float(x) if i==1 else int(x) for i,x in enumerate(temp[:-1])]
          row=head+temp
#          print(head)
#          print(row)
          data.append(row)
       else:
          if len(head)>0:
              head=[]
          head.append(int(temp[0][:-1]))
    f.close()
    return data


X = load_movie_ratings()

model = ALS()
model.fit(X, k=3, max_iter=6)

def writeto(file_name,contant):
    with open(file_name,"a+") as f:
        f.writelines(str(contant))
        f.writelines("\n")


def format(x):
    if x<1:
       x=1.0
    elif x>5:
       x=5.0
    else:
       x=round(x,0)
    return x

def predict_probe_ratings(model):
    maxmovieID=max(model.item_ids)
#    print(maxmovieID)
    file_name = "probe"
    path = os.path.join( "%s.txt" % file_name)
    f = open(path)
    lines = iter(f) 
    head=[]
    data=[]
    total_num=0
    mse0=0.00
    mse=0.00
    for line in lines:     
       temp=line[:-1].split(",")
       if temp[0][-1]==':':           
          if len(head)>0:
              head=[]
          if int(temp[0][:-1])>maxmovieID:
             continue
          writeto("predict",temp[0])
          head.append(int(temp[0][:-1]))
       else:
          if head==[]:
             continue
          temp=[int(temp[0])]
          prediction = model.predict(temp[0], head[0])
          if prediction is None:
             writeto("predict", prediction)
          else:
             total_num+=1
             writeto("predict",format(prediction[0]))
             truthpredict=model.ratings[temp[0]][head[0]]
             square_error=(format(prediction[0])- truthpredict)**2
             square_error0=(prediction[0]-truthpredict)**2
             mse0+=square_error0
             mse+=square_error
#          print(head)
    mse0=mse0/total_num
    mse=mse/total_num
#    print(total_num)
    rmse=mse**0.5
    rmse0=mse0**0.5
    f.close()
    print("probe集rmse为%.6f"%rmse0)
    print("probe集format后rmse为%.6f"%rmse)



predict_probe_ratings(model)


