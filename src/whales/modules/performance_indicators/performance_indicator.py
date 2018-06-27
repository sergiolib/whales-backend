from whales.modules.module import Module


class PerformanceIndicator(Module):
    def method_compute(self):
        raise NotImplementedError

    def compute(self):
        self.logger.info("Evaluating {}".format(self.__class__.__name__))
        return self.method_compute()
