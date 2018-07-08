from sklearn.externals import joblib


class SKLearnSaveLoadMixin:
    def method_load(self, location):
        self._model = joblib.load(location)

    def method_save(self, location):
        joblib.dump(self._model, location)
