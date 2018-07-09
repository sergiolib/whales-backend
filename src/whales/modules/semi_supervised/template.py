from whales.modules.semi_supervised.semi_supervised import SemiSupervised


class Demo(SemiSupervised):
    def __init__(self, logger=None):  # There should be no parameters here other that logger
        super().__init__(logger)  # This is necessary

        self.needs_fitting = False  # # If True, the method can't be used without fitting it
        # Also, if set, self.method_fit() is mandatory

        self.description = "This is a template"

        # Class attributes go here
        # Eg.:
        # self._model = DARR()

        self.parameters = {
            # Add parameters with default values here, as well as the inputs for training/predicting
        }

    def method_fit(self):
        # Here goes the training, depending on the method. Semi Supervised methods usually take data and target inputs
        # for training. Works if needs_fitting is True.
        pass

    def method_predict(self):
        # Here goes the prediction method. Semi Supervised methods usually take data and target input for predicting
        return None

    def method_load(self, location):
        # Here go the methods for loading models. Works if needs_fitting is True. Parameters are always already loaded
        # to a file ending with _parameters.json.
        pass

    def method_save(self, location):
        # Here go the methods for saving models. Works if needs_fitting is True. Parameters are always already saved to
        # a file ending with _parameters.json.
        pass


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
