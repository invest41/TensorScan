from flask import Flask,request, jsonify, render_template
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
model=load_model('model.h5')

#Loading the saved models in the ipynb file
app = Flask(__name__)

#Creating url routing for the flask web app
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    form=request.form['file']
    data_path=os.path.join(r'C:\Users\team\Desktop\copy\Tensorscan\examples',form)
    img=np.array(cv2.imread(data_path))
    scaled=img/255
    pred=int(model.predict(scaled.reshape(1,50,50,3)).round())
    to_str=['Cancer detected ' if pred==1 else 'Cancer not detected'][0]
    return render_template('predict.html',prediction_text=to_str,data_path=form)




#USE PIL
if __name__ == '__main__':
   app.run(debug=True)
