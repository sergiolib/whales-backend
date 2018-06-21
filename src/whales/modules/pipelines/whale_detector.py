from whales.modules.pipelines.pipeline import Pipeline, get_available_datasets


class WhaleDetector(Pipeline):
    def __init__(self, logger=None):
        super(WhaleDetector, self).__init__(logger)
        self.parameters = {
            "necessary_parameters": {
                "output_directory": str,
                "pipeline_type": str,
                "input_data": list,
                "input_labels": list,
            },
            "optional_parameters": {
                "pre_processing": list,
                "features_extractors": list,
                "performance_indicators": list,
                "machine_learning": dict,
                "data_set_type": dict,
                "active": bool,
                "verbose": bool,
                "seed": int,
            },
        }

    def initialize(self):
        self.load_input_data()
        self.load_labels()
        self.load_features_extractors()
        self.load_performance_indicators()

    def load_features_extractors(self):
        pass

    def load_performance_indicators(self):
        pass

    def instructions(self):
        execute_instruction = self.get_commands()
        while True:
            ins = self.next_instruction()
            if ins is None:
                break
            ins, param = ins
            param = {**self.results, **param}
            self.results = {**self.results, **(execute_instruction.get(ins, lambda x: {})(param))}


PipelineType = WhaleDetector
