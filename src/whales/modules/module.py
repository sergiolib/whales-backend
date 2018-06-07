import logging


class Module:
    """Generic module with inheritable methods"""
    def __init__(self, logger=logging.getLogger(__name__)):
        self.logger = logger

        self._parameters = {}

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict):
        for p, v in parameters.items():
            self._parameters[p] = v
