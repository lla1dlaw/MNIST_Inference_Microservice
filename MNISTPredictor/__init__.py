from flask import Flask
from MNISTPredictor.Model_Loader import Loader

app = Flask(__name__)
models = Loader("model_dicts")
