from whales.modules.pipelines.instructions_sets import SupervisedWhalesInstructionSet
from whales.modules.pipelines.loaders import TrainSupervisedWhalesDetectorLoaders
from whales.modules.pipelines.parsers import TrainWhalesPipelineParser
from whales.modules.pipelines.pipeline import Pipeline


class TrainWhaleDetector(Pipeline):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Training pipeline"

        self.loaders = TrainSupervisedWhalesDetectorLoaders(pipeline=self,
                                                            instructions_set=SupervisedWhalesInstructionSet(),
                                                            logger=self.logger)
        self.parser = TrainWhalesPipelineParser(logger=self.logger)

        self.parameters = {
            "input_data": [],
            "input_labels": [],
            "pre_processing": [],
            "features_extractors": [],
            "machine_learning": {},
            "verbose": True,
        }  # Default parameters for the API to serve

        self.private_parameters = {
            "results_directory": "",
            "logs_directory": "",
            "models_directory": "",
            "active": True,
            "seed": 0,
        }


PipelineType = TrainWhaleDetector
