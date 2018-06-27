import pandas as pd

from whales.modules.module import Module


class DataFile(Module):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.metadata = {}
        self._data = None
        self.file_name = None
        self.formatter = None

    def load_data(self, file_name: str, formatter):
        """Do not actually load the data. Instead, save the access information."""
        self.file_name = file_name
        self.formatter = formatter
        self.metadata = formatter.read_metadata(file_name)
        return self

    def save_data(self, file_name: str, formatter):
        """Save the data in self.data into specified file_name with specified formatter and also write the metadata.
        If no data has changed, read data from the Datafile and write it in the file."""
        formatter.write(file_name, self.data)
        formatter.write_metadata(file_name, self.metadata)

    def concatenate(self, datafiles_list, axis=1):
        """Add data_files from datafiles_list to new datafile and return it"""
        data = []
        new_df = self.__class__()
        metadata = {}
        for df in datafiles_list:
            # metadata[df.file_name] = df.metadata
            data_col = df.data.columns.drop("labels", errors="ignore")
            new_col = [f"data_{i}" for i, _ in enumerate(data_col)]
            df_data = df.data.rename(columns={a: b for a, b in zip(data_col.tolist(), new_col)})
            data.append(df_data)

        new_df.data = pd.concat(data, axis=axis)
        new_df.data.sort_index(inplace=True)
        new_df.metadata = metadata
        return new_df

    @property
    def data(self):
        if self._data is None:
            res = self.formatter.read(self.file_name)
        else:
            res = self._data
        return res

    @data.setter
    def data(self, data):
        self._data = data
