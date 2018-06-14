import numpy as np
from whales.modules.datafiles.datafiles import Datafile
from whales.modules.module import Module

assert_message_not_in_range = """{} proportion should be bigger than 0
and less than 1"""
assert_message_sum_bigger = """{} proportions sum {}.
No samples left for validation."""


class DataSet(Module):
    def __init__(self, logger=None):
        super(DataSet, self).__init__(logger)
        self.datafiles = []
        self.training_set = []
        self.testing_set = []
        self.validation_set = []
        self.training_set_inds = []
        self.testing_set_inds = []
        self.validation_set_inds = []

    def add_datafile(self, datafile: Datafile):
        self.datafiles.append(datafile)

    def remove_datafile(self, datafile: Datafile):
        self.datafiles.remove(datafile)

    def generate_3_sets(self, training_proportion=None,
                        testing_proportion=None, training_inds=None,
                        testing_inds=None):
        """Generate 3 sets: training, testing and validation. If inds not specified,
        sample them uniformly. validation_proportion is what's left from training and
        testing, so they can't sum more than 1.0"""

        if training_inds is not None and testing_inds is not None:
            self.training_set_inds = training_inds
            self.testing_set_inds = testing_inds

        elif training_proportion is not None and testing_proportion is not None:
            n = len(self.datafiles)
            all_inds = np.random.permutation(n)
            n_training = int(round(training_proportion * n))
            n_testing = int(round(testing_proportion * n))

            self.training_set_inds = all_inds[:n_training]
            self.testing_set_inds = all_inds[-n_testing:]

        else:
            self.logger.error("Incorrect set of parameters for generate_3_sets")
            raise RuntimeError

        self.training_set = [i for i in self.datafiles if i in self.training_set_inds]
        self.testing_set = [i for i in self.datafiles if i in self.testing_set_inds]
        self.validation_set = [i for i in self.datafiles
                               if i not in self.training_set_inds and i not in self.testing_set_inds]

    def generate_2_sets(self, training_proportion=None,
                        training_inds=None):
        """Generate 2 random sets: training and testing, sampled uniformly.
        testing_proportion is what's left from training, so it can't be more
        than 1.0"""
        if training_proportion is not None:
            return self.generate_3_sets(training_proportion=training_proportion,
                                        testing_proportion=1.0 - training_proportion)
        elif training_inds is not None:
            all_inds = list(range(len(self.datafiles)))
            return self.generate_3_sets(training_inds=training_inds,
                                        testing_inds=[i for i in all_inds if i not in training_inds])
