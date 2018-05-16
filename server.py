
# import Flask class from the flask module
from flask import Flask,request


import numpy as np
import pickle

# Create Flask object to run
app = Flask(__name__)

@app.route('/')
def home():
    return "Hi, Welcome to Flask!!"

@app.route('/predict')
def predict():

	# Get values from browser
	sepLen = request.args['sepal_length']
	sepWid = request.args['sepal_width']
	petLen = request.args['petal_length']
	petWid = request.args['petal_width']
	
	testData = np.array([sepLen, sepWid, petLen, petWid]).reshape(1,4)
	class_prediced = int(svmIrisModel.predict(testData)[0])
	output = "Predicted Iris Class: " + str(class_prediced)
	
	return (output)

def load_model():
	global svmIrisModel
	
	svmIrisFile = open('SVMModel.pckl', 'rb')
	svmIrisModel = pickle.load(svmIrisFile)
	svmIrisFile.close()	

if __name__ == "__main__":
	print("**Starting Server...")

# Call function that loads Model
	load_model()

	# Run Server
	app.run()
