from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipline import CustomData,PredictPipeline


application = Flask(__name__)

app = application

#creating route for homepage of app
@app.route('/')
def index():
    return render_template('index.html') # render_template will search the templates folder 
#routte for predicting and method which it support
@app.route('/prediction',methods = ['GET','POST'])
def predict_datapoint():
    # here we will do everything from gething the data make prediction
    if request.method =='GET':
        return render_template('home.html')
    # in post part we do bring the data, scaling and all
    else :
        # creat the data for that we make own customeData[the CustomeData class is created in prediction_pipline.py]
        data= CustomData(
            # we will try to read all the values 
            credit_score=int(request.form.get('credit_score')),
            country= request.form.get('country'),
            gender = request.form.get('gender'),
            age = int(request.form.get('age')),
            tenure = int(request.form.get('tenure')),
            balance=float(request.form.get('balance')),
             products_number = int(request.form.get('products_number')),
            credit_card=int(request.form.get('credit_card')),
            active_member=int(request.form.get('active_member')),
            estimated_salary=float(request.form.get('estimated_salary'))
            
        )
      
        # using function from customdata class to convert the data into datafram
        pred_df = data.get_data_as_data_frame()
        # if u want how your df look like print it 
        print(pred_df)

        # making object of PredictPipeline class present in prediction_pipline
        predict_pipline = PredictPipeline()
        # now give the i/p data and predict and store into results
        results = predict_pipline.predict(pred_df)
        return render_template('home.html', results = results[0]) #because it returning into list format and we read this result value in html to return the final prediction in frontend 
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)