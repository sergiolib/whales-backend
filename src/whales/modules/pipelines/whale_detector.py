from whales.modules.pipelines.instruction_sets import SupervisedWhalesInstructionSet
from whales.modules.pipelines.loaders import SupervisedWhalesDetectorLoaders
from whales.modules.pipelines.parsers import WhalesPipelineParser
from whales.modules.pipelines.pipeline import Pipeline


class WhaleDetector(Pipeline):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Whale Detector Pipeline"

        self.loaders = SupervisedWhalesDetectorLoaders(pipeline=self, instructions_set=SupervisedWhalesInstructionSet())
        self.parser = WhalesPipelineParser()

        self.parameters = {}


PipelineType = WhaleDetector
