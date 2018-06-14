import logging


class Module:
    """Generic module with inheritable methods"""
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(str(self))
        self.description = ""

        self._parameters = {}

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict):
        for p, v in parameters.items():
            self._parameters[p] = v

    @property
    def short_name(self):
        class_name = self.__class__.__name__
        return "".join([i for i in class_name if i.isdigit() or i.isupper()])

    def __str__(self):
        return self.__class__.__name__