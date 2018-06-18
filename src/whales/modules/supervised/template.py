from whales.modules.supervised.supervised import Supervised


class Demo(Supervised):
    def __init__(self, logger=None):  # There should be no parameters here
        super(Demo, self).__init__(logger)

        self.description = "This is a demo"

        # Class attributes go here
        # Eg.:
        # self._model = sklearn.LogisticRegression

        self.parameters = {
            # Add parameters here
        }

    def method_fit(self, data, target):
        # Here goes the training. Supervised takes data and target inputs for training
        pass

    def method_predict(self, data):
        # Here goes the training. Unsupervised takes only data input for predicting
        return None


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
