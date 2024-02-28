def predict():
	msg_data = {}
	for k in request.args.keys():
		val=request.args.get(k)
		msg_data[k]=val
	f = open("models/X_test.json","r")
	X_test = json.load(f)
	f.close()
	all_cols = X_test
	input_df = pd.DataFrame(msg_data,columns=all_cols,index=[0])
	model = pickle.load(open(data, "rb"))
	arr_results = model.predict(input_df)
	diabetes_likelihood = ""
	if arr_results[0] == 0:
		diabetes_likelihood = "No"
	elif arr_results[0] == 1:
		diabetes_likelihood = "Yes"
	return diabetes_likelihood
