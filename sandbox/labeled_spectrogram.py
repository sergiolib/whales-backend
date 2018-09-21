from whales.modules.data_files.audio import AudioDataFile
from whales.modules.formatters.aif import AIFFormatter
from whales.modules.labels_formatters.csv import CSVLabelsFormatter
from whales.modules.performance_indicators.labeled_spectrogram import LabeledSpectrogram
import matplotlib.pyplot as plt

df = AudioDataFile().load(
    "/home/sliberman/Dropbox/Detector ballena azul/supervised_version/database/validation-ballenas/ballenas-S06_PU145_20120514_143000.aif",
    formatter=AIFFormatter())
df.load_labels(
    "/home/sliberman/Dropbox/Detector ballena azul/supervised_version/database/etiquetas/csv/Chile01_002K_S06_PU145_20120514_143000-Labels.csv",
    labels_formatter=CSVLabelsFormatter())
ls = LabeledSpectrogram()
ls.private_parameters["data_file"] = df
ls.method_compute()
plt.show()
