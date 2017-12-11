import lstm, os, sys

#Step 1 Load Data

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR) + "/tools/")

X_train, y_train, X_test, y_test = lstm.load_data(CURRENT_DIR + '\\btc.csv', 50, True)

#Step 2 Build Model
layers = [1, 50, 100, 1] # input dimension LSTM, output dimension LSTM, secound LST output dimension, output dimension dense
model = lstm.build_model(layers)

#Step 3 Train the model
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05)

#Step 4 - Plot the predictions!
predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)