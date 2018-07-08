from whales.modules.pipelines.instructions_sets import SupervisedWhalesInstructionSet
from whales.modules.pipelines.loaders import TrainSupervisedWhalesDetectorLoaders
from whales.modules.pipelines.parsers import TrainWhalesPipelineParser
from whales.modules.pipelines.pipeline import Pipeline


class TrainWhaleDetector(Pipeline):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Pipeline for training a whale detector"

        self.loaders = TrainSupervisedWhalesDetectorLoaders(pipeline=self,
                                                            instructions_set=SupervisedWhalesInstructionSet())
        self.parser = TrainWhalesPipelineParser()

        self.parameters = {}


PipelineType = TrainWhaleDetector
