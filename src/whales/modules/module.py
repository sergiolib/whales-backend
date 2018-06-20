from os.path import splitext, join

import logging
import json


class Module:
    """Generic module with inheritable methods"""
    description = ""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(str(self))
        self.needs_fitting = False  # Default
        self.is_fitted = False  # Default
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

    def save_parameters(self, location):
        """Save parameters to disk"""
        json.dump(self.parameters, open(location, 'w'))

    def load_parameters(self, location):
        """Load parameters from disk"""
        self.parameters = json.load(open(location, 'r'))

    def save(self, location):
        """Save module and its parameters"""
        loc, ext = splitext(location)
        params_location = join(loc, "_parameters.json")
        self.save_parameters(params_location)
        if self.needs_fitting and self.is_fitted:
            self.method_save(location)

    def load(self, location):
        """Load module and its parameters"""
        loc, ext = splitext(location)
        params_location = join(loc, "_parameters.json")
        self.load_parameters(params_location)
        if self.needs_fitting:
            self.method_load(location)
            self.is_fitted = True
