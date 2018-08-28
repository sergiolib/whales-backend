from whales.modules.pipelines.instructions_sets import SupervisedWhalesInstructionSet
from whales.modules.pipelines.loaders import SupervisedWhalesDetectorLoaders
from whales.modules.pipelines.parsers import WhalesPipelineParser
from whales.modules.pipelines.pipeline import Pipeline


class WhaleDetectorForTests(Pipeline):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Whale Detector Pipeline that trains and predicts"

        self.loaders = SupervisedWhalesDetectorLoaders(pipeline=self, instructions_set=SupervisedWhalesInstructionSet(),
                                                       logger=self.logger)
        self.parser = WhalesPipelineParser(logger=self.logger)

        self.parameters = {}


# PipelineType = WhaleDetectorForTests
