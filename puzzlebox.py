import pandas as pd
from pycaret import classification


class PuzzleBox:
    def __init__(self, input_file, model_name):
        self.data = pd.read_csv(input_file)
        self.model_name = model_name

    def create_model(self, model, target):
        classification.setup(data=self.data, target=target, silent=True, html=False)
        return classification.create_model(model)

    @staticmethod
    def plot_results(model, plot):
        classification.plot_model(model, plot=plot)

    @staticmethod
    def evaluate(model):
        classification.evaluate_model(model)

    @staticmethod
    def interpret(model):
        classification.interpret_model(model)

    @staticmethod
    def predict(model, data):
        classification.predict_model(model, data=pd.read_csv(data))

    def save(self, model):
        classification.save_model(model, self.model_name)

    def load(self):
        return classification.load_model(model_name=self.model_name)


