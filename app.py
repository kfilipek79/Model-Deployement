
import numpy as np
from flask import Flask,jsonify,request

import pickle

pickle_rfc=pickle.load(open('rfc_model.pkl','rb'))

app=Flask(__name__)


@app.route('/ml_api',methods=['POST'])
def predict():
	input_data=request.get_json(force=True)

	input_feature=[input_data['a'],input_data['b'],input_data['c'],input_data['d']]

	input_feature=np.array(input_feature).reshape(1,4)
	pred_class=pickle_rfc.predict(input_feature)
	
	final_class=pred_class[0]
	return jsonify(results=final_class)


if __name__=='__main__':
	app.run(port=9999,debug=True)

