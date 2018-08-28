from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Demo(FeatureExtraction):
    def __init__(self, logger=None):  # There should be no parameters here other than logger
        super().__init__(logger)  # Always call super().__init__(logger) like this

        self.needs_fitting = True  # If True, the method can't be used without fitting it
        # Also, if set, self.method_fit() is mandatory

        self.description = "This is a template"

        # Attributes go here
        # Eg.:
        # self._model = Model()

        self.parameters = {
            # Add parameters and default values
        }

    def method_fit(self):
        data = self.all_parameters["data"]  # Get data and maybe other parameters
        # Here goes the training, if needed
        pass

    def method_transform(self):
        data = self.all_parameters["data"]  # Get data and maybe other parameters
        # Here goes the transforms
        return None

    def method_load(self, location):
        # Here goes the method for loading models. Works if needs_fitting is True. Parameters are always already loaded
        # to a file ending with _parameters.json
        pass

    def method_save(self, location):
        # Here goes the method for saving models. Works if needs_fitting is True. Parameters are always already saved to
        # a file ending with _parameters.json
        pass


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
