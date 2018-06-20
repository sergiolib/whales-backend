import pandas as pd
from whales.modules.labels_formatters.labels_formatters import LabelsFormatter


class CSVLabelsFormatter(LabelsFormatter):
    def __init__(self, logger=None):
        super(CSVLabelsFormatter, self).__init__(logger)
        self.description = """CSV label files parser and loader"""

    def read(self, filename):
        read_file = pd.read_csv(filename, index_col=None, header=0)
        return read_file


PipelineFormatter = CSVLabelsFormatter
