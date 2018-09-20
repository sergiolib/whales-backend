from whales.modules.data_files.feature import AudioSegments
from whales.modules.data_sets.data_sets import DataSet


class NoSplit(DataSet):
    """Generate just one data set"""
    def __init__(self, logger=None):
        super().__init__(logger)
        self.current_file = 0

        self.description = """1 set, no splitting"""

        self.parameters = {}

    @property
    def iterations(self):
        return 1 if len(self.datafiles) > 0 else 0

    def get_data_sets(self):
        tr = AudioSegments()
        tr.metadata["sampling_rate"] = {}
        for i, vi in enumerate(self.datafiles):
            for j in range(vi.all_parameters["number_of_windows"]):
                tr.label_name.update(vi.label_name)
                window, label = vi.get_window(j)
                window.name = f"{i}_{j}"
                tr.add_segment(window, label)
                tr.metadata["sampling_rate"][window.name] = vi.sampling_rate
        yield tr


PipelineDataSet = NoSplit
