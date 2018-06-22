from whales.modules.unsupervised.unsupervised import Unsupervised


class Demo(Unsupervised):
    def __init__(self, logger=None):  # There should be no parameters here
        super().__init__(logger)

        self.needs_fitting = True  # If True, the method can't be used without fitting it.
        # Also, if set, Demo().fit() will be callable. Else, it won't.

        self.description = "This is a demo"

        # Class attributes go here
        # Eg.:
        # self._model = sklearn.DBScan

        self.parameters = {
            # Add parameters here
        }

    def method_fit(self, data):
        # Here goes the training. Unsupervised takes only data input for training. Works only if needs_fitting is True
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
