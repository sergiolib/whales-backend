import json
from whales.modules.formatters.formatters import Formatter


class JSONFormatter(Formatter):
    def read(self, filename):
        return json.load(open(filename, mode="r"))

    def write(self, filename, data):
        json.dump(data, open(filename, mode="w"))
