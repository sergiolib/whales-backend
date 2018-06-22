from whales.modules.pipelines.instruction_sets import SupervisedWhalesInstructionSet
from whales.modules.pipelines.loaders import SupervisedWhalesDetectorLoaders
from whales.modules.pipelines.parsers import PipelineParser
from whales.modules.pipelines.pipeline import Pipeline


class WhaleDetector(Pipeline):
    def __init__(self, logger=None):
        super(WhaleDetector, self).__init__(logger)

        self.instruction_set = SupervisedWhalesInstructionSet()
        self.loaders = SupervisedWhalesDetectorLoaders(pipeline=self)
        self.parser = PipelineParser(pipeline=self)

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


PipelineType = WhaleDetector
