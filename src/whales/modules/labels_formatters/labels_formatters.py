from whales.modules.module import Module


class LabelsFormatter(Module):
    def read(self, filename):
        raise NotImplementedError

    def write(self, filename, data):
        raise NotImplementedError
