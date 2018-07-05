from whales.utilities.testing import get_5_file_names, get_labeled
from whales.modules.data_files.audio import AudioDataFile
from whales.modules.formatters.aif import AIFFormatter
from whales.modules.labels_formatters.csv import CSVLabelsFormatter
from whales.modules.supervised.svm import SVM


# def test_svm():
#     data, labels = get_labeled()
#     df = AudioDatafile().load(data, formatter=AIFFormatter)
#     df.load_labels(labels, labels_formatter=CSVLabelsFormatter, label="whale")
#     X = df.data.drop("labels", axis=1).values
#     y = df.get_labeled_data().labels.values
#     # SVM().fit(Xf, y)
