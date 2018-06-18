import pandas as pd
from whales.modules.labels_formatters.labels_formatters import LabelsFormatter


class TXTLabelsFormatter(LabelsFormatter):
    def read(filename):
        read_file = pd.read_csv(filename, index_col=None, header=0, sep="\t+")
        return read_file

PipelineFormatter = TXTLabelsFormatter
