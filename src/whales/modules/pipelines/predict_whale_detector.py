from whales.modules.pipelines.instructions_sets import SupervisedWhalesInstructionSet
from whales.modules.pipelines.loaders import PredictSupervisedWhalesDetectorLoaders
from whales.modules.pipelines.parsers import PredictWhalesPipelineParser
from whales.modules.pipelines.pipeline import Pipeline


class PredictWhaleDetector(Pipeline):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Predictions pipeline"

        self.loaders = PredictSupervisedWhalesDetectorLoaders(pipeline=self,
                                                              instructions_set=SupervisedWhalesInstructionSet())
        self.parser = PredictWhalesPipelineParser()

        self.parameters = {
            "output_directory": "",
            "input_data": [],
            "input_labels": [],
            "pre_processing": [],
            "features_extractors": [],
            "performance_indicators": [],
            "machine_learning": {},
            "active": True,
            "verbose": False,
            "seed": 0,
        }  # Default parameters for the API to serve


PipelineType = PredictWhaleDetector
