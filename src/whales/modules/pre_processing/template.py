from whales.modules.pre_processing.pre_processing import PreProcessing


class Demo(PreProcessing):
    def __init__(self, logger=None):  # There should be no parameters here
        super().__init__(logger)

        self.description = "This is a demo"

        # Class attributes go here
        # Eg.:
        # self._model = Scale()

        self.parameters = {
            # Add parameters here
        }

    def method_transform(self, data_file):
        # Here goes the training. Same parameters as in fit
        return None


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
