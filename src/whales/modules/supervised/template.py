from whales.modules.supervised.supervised import Supervised


class Demo(Supervised):
    def __init__(self, logger=None):  # There should be no parameters here
        super(Demo, self).__init__(logger)

        self.needs_fitting = True  # If True, the method can't be used without fitting it
        # Also, if set, Demo().fit() will be callable. Else, it won't.

        self.description = "This is a demo"

        # Class attributes go here
        # Eg.:
        # self._model = sklearn.LogisticRegression

        self.parameters = {
            # Add parameters here
        }

    def method_fit(self, data, target):
        # Here goes the training. Supervised takes data and target inputs for training. Works if needs_fitting is True.
        pass

    def method_predict(self, data):
        # Here goes the training. Unsupervised takes only data input for predicting
        return None

    def method_load(self, location):
        # Here go the methods for loading models. Parameters are always already loaded to a file ending with
        # *_parameters.json. Works if needs_fitting is True.
        pass

    def method_save(self, location):
        # Here go the methods for saving models. Parameters are always already saved to a file ending with
        # *_parameters.json. Works if needs_fitting is True.
        pass


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
