from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

class SignLanguageModel():
    def __init__(self):
        self.model = Sequential()

        # input layer
        self.model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(60,150)))

        #hidden layers
        self.model.add(LSTM(256, return_sequences=True, activation='relu'))
        self.model.add(LSTM(256, return_sequences=True, activation='relu'))
        self.model.add(LSTM(128, return_sequences=False, activation='relu'))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))

        #output layer
        #Only 5 words are used
        self.model.add(Dense(5, activation='softmax'))

        # Loading a saved model
        self.model.load_weights("weights.h5")

    def pre_process(self, input_data, req):
        return input_data['features']

    def predict(self, processed_data, req):
        return self.model.predict_proba(processed_data)[0]

    def post_process(self, prediction, req):
        return {'probability': prediction}
