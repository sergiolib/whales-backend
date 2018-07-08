from whales.modules.pipelines.instructions_sets import SupervisedWhalesInstructionSet
from whales.modules.pipelines.loaders import PredictSupervisedWhalesDetectorLoaders
from whales.modules.pipelines.parsers import PredictWhalesPipelineParser
from whales.modules.pipelines.pipeline import Pipeline


class PredictWhaleDetector(Pipeline):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Pipeline for training a whale detector"

        self.loaders = PredictSupervisedWhalesDetectorLoaders(pipeline=self,
                                                              instructions_set=SupervisedWhalesInstructionSet())
        self.parser = PredictWhalesPipelineParser()

        self.parameters = {}


PipelineType = PredictWhaleDetector
