from whales.modules.unsupervised.unsupervised import Unsupervised


class Demo(Unsupervised):
    def __init__(self, logger=None):  # There should be no parameters here
        super(Demo, self).__init__(logger)

        self.description = "This is a demo"

        # Class attributes go here
        # Eg.:
        # self._model = sklearn.DBScan

        self.parameters = {
            # Add parameters here
        }

    def method_fit(self, data):
        # Here goes the training. Unsupervised takes only data input for training
        pass

    def method_predict(self, data):
        # Here goes the training. Unsupervised takes only data input for predicting
        return None


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
