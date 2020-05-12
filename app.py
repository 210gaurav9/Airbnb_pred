import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from flask import Flask, request, render_template
import xgboost as xgb

app = Flask(__name__)
#model=joblib.load('airbnb1.pkl')
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('airbnb.model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    features=pd.DataFrame(features)
    features[0][11] = pd.to_datetime(features[0][11],infer_datetime_format=True)
    features=features.values.reshape(1,-1)
    features=pd.DataFrame(features)
    
    b="2011-03-28"
    b=pd.to_datetime(b)
    features[11] = features[11].apply(lambda x: x.toordinal() - b.toordinal())

    num=features[4].str.slice(stop=1)
    num=int(num)
    num=num-1
    for i in range(5):
        if i==num:
            features[15+i]=1
        else:
            features[15+i]=0
    #c=0
    num1=0
    g=features[5].to_string()
    g=g.split(".")
    num1=int(g[0][1:])
    #for i in range(6,len(g)):
     #   if g[5:i].isdigit():
      #      c=c+1;
       # else:
        #    break
    #num1=int(g[5:(c+5)])
    num1=num1-1
    for i in range(223):
        if i==num1:
            features[20+i]=1
        else:
            features[20+i]=0

    num2=features[8].str.slice(stop=1)
    num2=int(num2)
    num2=num2-1
    for i in range(4):
        if i==num2:
            features[242+i]=1
        else:
            features[242+i]=0

    num3=int(features[14])
    if num3 > 353:
        features[245]=1
    else:
        features[245]=0
    if num3 < 12:
        features[246]=1
    else:
        features[246]=0
    num4=float(features[12])
    if num4 == 0.0:
        features[247]=1
    else:
        features[247]=0

    features.drop([0,1,2,3,4,5,8], inplace=True, axis=1)
    features[6]=float(features[6])
    features[7]=float(features[7])
    features[9]=int(features[9])
    features[10]=int(features[10])
    features[12]=float(features[12])
    features[13]=int(features[13])
    features[14]=int(features[14])
    
    #scaler = RobustScaler()
    #Xpred = scaler.fit_transform(features)
    var=xgb.DMatrix(features)
    #final_features = np.array(float_features)
    #final_features[0]=final_features[0]*365
    #df=final_features.reshape(1,-1)
    

    prediction = bst.predict(var)
    prediction = np.exp(prediction)-1
    #output = np.round(prediction)

    return render_template('output.html', prediction_text='${}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)