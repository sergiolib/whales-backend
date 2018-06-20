from sklearn.externals import joblib


class SKLearnSaveLoadMixin:
    def __init__(self):
        self._method = None

    def method_load(self, location):
        self._method = joblib.load(location)

    def method_save(self, location):
        joblib.dump(self._method, location)
