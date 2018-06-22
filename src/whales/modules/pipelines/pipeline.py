"""Package to parse a parameters file and execute what the user requests"""

import json
import logging
from multiprocessing import Process
from whales.modules.module import Module


class Pipeline(Module):
    """Module that parses JSON formatted parameter set, sets up the pipelines and allows launching it.
    Actual pipelines initialization or instructions are implemented in children classes."""
    def __init__(self, logger=logging.getLogger(__name__)):
        super(Pipeline, self).__init__(logger)
        self.process = None
        self.description = "Generic Pipeline"
        self.instructions_series = []
        self.loaders = None
        self.parser = None
        self.results = {}

        # Default parameters
        self.parameters = {}

    def start(self):
        """Run the instructions series"""
        self.logger.debug(f"Pipeline started")
        # self.process = Process(target=self.instructions)
        # self.process.start()
        self.instructions()

    def load_parameters(self, parameters_file: str):
        dictionary = json.loads(parameters_file)
        self.parameters = self.parser.parse(dictionary)
        self.logger.debug("Parameters set")

    def instructions(self):
        while True:
            ins = self.next_instruction()
            if ins is None:
                break  # Finished executing instructions. Results in the results attribute
            fun, param = ins
            param["results"] = self.results
            if fun is None:
                raise RuntimeError(f"Instruction {ins} is not defined")
            instruction_results = fun(param)
            if type(instruction_results) is not dict:
                raise RuntimeError(f"Instruction {ins} should return a dictionary")
            self.results = {**self.results, **instruction_results}

    def add_instruction(self, instruction_type, instruction_parameters):
        """Adds instruction to the last execution place"""
        self.instructions_series.append((instruction_type, instruction_parameters))

    def next_instruction(self):
        """Returns the next instruction"""
        if len(self.instructions_series) == 0:
            instruction = None
        else:
            instruction = self.instructions_series.pop(0)
        return instruction

    def initialize(self):
        for loader in self.loaders.loaders_execution_order:
            loader()
