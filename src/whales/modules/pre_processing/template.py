from whales.modules.pre_processing.pre_processing import PreProcessing


class Demo(PreProcessing):
    def __init__(self, logger=None):  # There should be no parameters here other that logger
        super().__init__(logger)  # Always call super().__init__(logger) like this

        self.description = "This is a template"

        # Class attributes go here
        # Eg.:
        # self._model = Scale()

        self.parameters = {
            # Add parameters here and default values
        }

    def method_transform(self):
        # data_file = self.parameters["data"]  # data_file is an AudioSegments data file
        # Here goes the transformation. Same parameters as in fit
        # Should return a numpy array of NxD where N is the number of rows and D the number of columns
        return None


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
