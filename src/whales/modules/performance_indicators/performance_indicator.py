from whales.modules.module import Module


class PerformanceIndicator(Module):
    def method_evaluate(self):
        raise NotImplementedError

    def evaluate(self):
        self.logger.info("Evaluating {}".format(self.__name__))
        return self.method_evaluate()
