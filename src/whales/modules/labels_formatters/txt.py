import pandas as pd
from whales.modules.labels_formatters.labels_formatters import LabelsFormatter


class TXTLabelsFormatter(LabelsFormatter):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = """TXT label files parser and loader"""

    def read(self, filename):
        read_file = pd.read_csv(filename, index_col=None, header=0, sep="\t+", engine="python")
        return read_file


PipelineFormatter = TXTLabelsFormatter
