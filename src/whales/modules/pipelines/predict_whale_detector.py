from whales.modules.pipelines.instructions_sets import WhalesInstructionSet
from whales.modules.pipelines.loaders import PredictSupervisedWhalesDetectorLoaders
from whales.modules.pipelines.parsers import PredictWhalesPipelineParser
from whales.modules.pipelines.pipeline import Pipeline


class PredictWhaleDetector(Pipeline):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Predictions pipeline"

        self.loaders = PredictSupervisedWhalesDetectorLoaders(pipeline=self,
                                                              instructions_set=WhalesInstructionSet(),
                                                              logger=logger)
        self.parser = PredictWhalesPipelineParser(logger=self.logger)

        self.parameters = {
            "input_data": [],
            "input_labels": [],
            "pre_processing": [],
            "features_extractors": [],
            "performance_indicators": [],
            "machine_learning": {},
            "verbose": False,
        }  # Default parameters for the API to serve

        self.private_parameters = {
            "results_directory": "",
            "logs_directory": "",
            "models_directory": "",
            "active": True,
            "seed": 0,
        }


PipelineType = PredictWhaleDetector
