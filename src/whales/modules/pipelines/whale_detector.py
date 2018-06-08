from whales.modules.pipelines.pipeline import Pipeline


class WhaleDetector(Pipeline):
    def initialize(self):
        mandatory_parameters = {
            "feature_extractors": list,  # List of feature extractors
            "performance_indicators": list,  # List of performance indicators
            "input_data": str  # Path to data file
        }
        self.check_parameters(mandatory_parameters)

        self.load_input_data()
        self.load_features_extractors()
        self.load_performance_indicators()

    def check_parameters(self, mandatory_parameters: dict):
        """Check if parameters loaded with method load_parameters are present and are of the expected type"""
        for p, expected_type in mandatory_parameters.items():
            actual_type = type(self.parameters[p])
            if p not in self.parameters:
                raise AttributeError(f"Mandatory parameter {p} was not found")
            elif actual_type is not expected_type:
                raise AttributeError(f"Mandatory parameter {p} is {actual_type} and it should be {expected_type}")

    def load_input_data(self):
        pass

    def load_features_extractors(self):
        pass

    def load_performance_indicators(self):
        pass


PipelineType = WhaleDetector
