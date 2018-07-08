"""Package to parse a parameters file and execute what the user requests"""

import json
import logging
from multiprocessing import Process
from whales.modules.module import Module
import pprint


class Pipeline(Module):
    """Module that parses JSON formatted parameter set, sets up the pipelines and allows launching it.
    Actual pipelines initialization or instructions are implemented in children classes."""
    def __init__(self, logger=logging.getLogger(__name__)):
        super().__init__(logger)
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
        self.logger.info(f"{self.__class__.__name__} pipeline started")
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
            param.update(self.results)
            if fun is None:
                raise RuntimeError(f"Instruction {fun} is not defined")
            instruction_results = fun(param)
            if type(instruction_results) is not dict:
                raise RuntimeError(f"Instruction {fun} should return a dictionary")
            self.results.update(instruction_results)
        return self.results

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

    def __repr__(self):
        ret = []
        for i, r in enumerate(self.instructions_series):
            ret.append("\n".join([str(i + 1) + ". " + "\033[1m" + r[0].__name__ + "\033[0m", pprint.pformat(r[1])]))
        return "\n\n".join(ret)
