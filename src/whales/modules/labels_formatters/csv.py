import pandas as pd
from whales.modules.labels_formatters.labels_formatters import LabelsFormatter


class CSVLabelsFormatter(LabelsFormatter):
    def read(filename):
        read_file = pd.read_csv(filename, index_col=None, header=0)
        return read_file
