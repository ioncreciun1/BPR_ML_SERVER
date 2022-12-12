from flask import Flask
from flask import request
import json
from signLanguageModel import SignLanguageModel
from flask_cors import CORS,cross_origin


app = Flask(__name__)
CORS(app)


@cross_origin()
@app.route("/sign-language-translation", methods = ["POST"])
def get_word():
  data = json.loads(request.data)
  preprocessed_data = model.get_preprocess_data(data)
  word = model.predict(preprocessed_data)
  print(word)
  return {"predicted_word":word}

if __name__ == "__main__":
  model = SignLanguageModel()
  app.run( port=80)