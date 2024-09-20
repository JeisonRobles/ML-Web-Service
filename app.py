import pickle
import numpy as np
from flask import Flask, request, jsonify

path = '/Users/jeisonroblesarias/Documents/GitHub/APIa/'


with open(path + 'linear_regression_model.pkl', 'rb') as f:
    model, X_train, y_train= pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    if 'input' not in data:
        return jsonify({'error': 'Input is required'}), 400

    prediction = model.predict(np.array([[data['input']]]))

    return jsonify({f'The predicted value for input is:':prediction[0]})


@app.route('/retrain', methods=['POST'])
def retrain():

    global X_train, y_train

    data = request.get_json(force=True)

    input_value = float(data['input'])
    target_value = float(data['target'])

    new_X = np.array([[input_value]])
    new_y = np.array([target_value])

    X_train = np.vstack((X_train,new_X))
    y_train = np.concatenate((y_train,new_y))

    model.fit(X_train,y_train)

    with open(path + 'linear_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return jsonify({'Message': 'Model Retrained Successfully'})


if __name__ == '__main__':
    app.run(debug=True)

