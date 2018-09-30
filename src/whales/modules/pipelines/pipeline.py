"""Package to parse a parameters file and execute what the user requests"""

import json
import logging
from multiprocessing import Process
from os import makedirs
from os.path import join

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

    def load_parameters_from_json(self, json_string: str):
        dictionary = json.loads(json_string)
        self.load_parameters_from_dict(dictionary)

    def load_parameters_from_dict(self, dictionary):
        self.parameters = self.parser.parse(dictionary)
        self.logger.debug("Parameters set")

    def load_parameters(self, arg):
        if type(arg) is str:
            self.load_parameters_from_json(arg)
        elif type(arg) is dict:
            self.load_parameters_from_dict(arg)

        # Set up logger
        if "logs_directory" in self.all_parameters:
            makedirs(self.all_parameters["logs_directory"], exist_ok=True)
            logger_path = join(self.all_parameters["logs_directory"], "messages.log")
            hdlr = logging.FileHandler(logger_path, mode="w")
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            hdlr.setFormatter(fmt)
            self.logger = logging.Logger(name=str(self))
            self.logger.addHandler(hdlr)
            if self.all_parameters.get("verbose", False):
                self.logger.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(logging.INFO)

            self.loaders.logger = self.logger
            self.loaders.instructions_set.logger = self.logger

        self.logger.info("Parameters loaded correctly")

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
        self.logger.info("Pipeline initialized")

    def __repr__(self):
        ret = []
        for i, r in enumerate(self.instructions_series):
            ret.append("\n".join([str(i + 1) + ". " + "\033[1m" + r[0].__name__ + "\033[0m", pprint.pformat(r[1])]))
        return "\n\n".join(ret)
