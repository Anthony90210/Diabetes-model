import pandas as pd
import json
import pickle
from flask import Flask, render_template, request
from werkzeug.serving import run_simple


# WEBAPP
app = Flask(__name__)

# # Save model using joblib
# joblib.dump(voting_model, 'model.joblib')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'dataset_file' not in request.files:
        return 'No file uploaded', 400

    # Get the uploaded file
    dataset_file = request.files['dataset_file']

    # Check if the file is a CSV file
    if dataset_file.filename.endswith('.csv'):
        render_template('index.html')

        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(dataset_file)

        # msg_data = {}
        # for k in request.args.keys():
        #     val = request.args.get(k)
        #     msg_data[k] = val
        # f = open("models/X_test.json", "r")
        #
        # x_test = json.load(f)
        # f.close()
        # all_cols = x_test

        # input_df = pd.DataFrame(msg_data, columns=all_cols, index=[0])
        model_name = "models/model.pkl"
        # load model
        model = pickle.load(open(model_name, "rb"))
        arr_results = model.predict(data)

        diabetes_likelihood = ""
        if arr_results[0] == 0:
            diabetes_likelihood = "NO - Based on the provided data probability of diabetes is 0"
        elif arr_results[0] == 1:
            diabetes_likelihood = "Yes - Based on the provided data probability of diabetes is 1"
        return diabetes_likelihood

    else:
        return 'Invalid file format. Please upload a CSV file.', 400


if __name__ == '__main__':
    app.run(debug=True)
    run_simple("localhost", 5000, app)
