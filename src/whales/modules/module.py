from os.path import splitext, join

import logging
import json

import pickle

from whales.utilities.json import WhalesEncoder, WhalesDecoder


class Module:
    """Generic module with inheritable methods"""
    description = ""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(str(self))
        self.needs_fitting = False  # Default
        self.is_fitted = False  # Default
        self._public_parameters = {}
        self._private_parameters = {}
        self._parameters_options = {}

    @property
    def parameters(self):
        return self._public_parameters

    @parameters.setter
    def parameters(self, parameters: dict):
        self._public_parameters.update(parameters)

    @property
    def private_parameters(self):
        return self._private_parameters

    @private_parameters.setter
    def private_parameters(self, parameters: dict):
        self._private_parameters.update(parameters)

    @property
    def all_parameters(self):
        return {**self.private_parameters, **self.parameters}

    @property
    def parameters_options(self):
        return self._parameters_options

    @parameters_options.setter
    def parameters_options(self, options: dict):
        self._parameters_options.update(options)

    @property
    def short_name(self):
        class_name = self.__class__.__name__
        return "".join([i for i in class_name if i.isdigit() or i.isupper()])

    def __str__(self):
        return self.__class__.__name__

    def save_parameters(self, location):
        """Save parameters to disk"""
        parameters = self.parameters.copy()
        parameters["module_type"] = str(self)
        json.dump(parameters, open(location, 'w'), cls=WhalesEncoder)

    def load_parameters(self, location):
        """Load parameters from disk"""
        self.parameters = json.load(open(location, 'r'), cls=WhalesDecoder)
        if not self.all_parameters["module_type"] == str(self):
            raise ValueError(f"Impossible to load a {self.all_parameters['module_type']} into a {str(self)} module")
        del self.all_parameters["module_type"]

    def save(self, location):
        """Save module and its parameters"""
        loc, ext = splitext(location)
        params_location = loc + "_parameters.json"
        self.save_parameters(params_location)
        if self.needs_fitting and self.is_fitted:
            self.method_save(location)

    def load(self, location):
        """Load module and its parameters"""
        loc, ext = splitext(location)
        params_location = loc + "_parameters.json"
        self.load_parameters(params_location)
        if self.needs_fitting:
            self.method_load(location)
            self.is_fitted = True

    def method_save(self, location):
        raise NotImplementedError

    def method_load(self, location):
        raise NotImplementedError

    def __repr__(self):
        name = self.__class__.__name__
        params = str(self.parameters)
        return " ".join([name, params])
