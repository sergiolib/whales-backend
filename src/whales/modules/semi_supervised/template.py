from whales.modules.semi_supervised.semi_supervised import SemiSupervised


class Demo(SemiSupervised):
    def __init__(self, logger=None):  # There should be no parameters here
        super(Demo, self).__init__(logger)

        self.description = "This is a demo"

        # Class attributes go here
        # Eg.:
        # self._model = SomeGraph()

        self.parameters = {
            # Add parameters here
        }

    def method_fit(self, data, target):
        # Here goes the training, depending on the method. Semi Supervised takes data and target inputs for training
        pass

    def method_predict(self, data, target):
        # Here goes the training. Semi Supervised takes data and target input for predicting
        return None


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
