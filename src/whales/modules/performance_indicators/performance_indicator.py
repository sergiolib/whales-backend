from os.path import splitext

import numpy as np
from matplotlib.pyplot import Figure
from whales.modules.module import Module


class PerformanceIndicator(Module):
    def method_save_results(self, location: str):
        loc, ext = splitext(location)
        res = self.method_compute()
        file_type = None
        if type(res) in [int, float]:
            location = loc + ".txt"
            with open(location, mode='w') as destination:
                destination.write(str(res))
            file_type = "scalar"
        elif type(res) is np.ndarray:
            location = loc + ".txt"
            np.savetxt(location, res)
            file_type = "array"
        elif type(res) is Figure:
            location = loc + ".png"
            res.savefig(location)
            file_type = "graphic"
        else:
            raise NotImplementedError
        return location, file_type

    def method_compute(self):
        raise NotImplementedError

    def compute(self):
        self.logger.info("Evaluating performance indicator {}".format(self.__class__.__name__))
        return self.method_compute()

    def save_results(self, location: str):
        location, file_type = self.method_save_results(location)
        self.logger.info(f"Saved performance indicator {self} into {location} as a {file_type}")
