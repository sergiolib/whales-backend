from whales.modules.datafiles.datafiles import Datafile


class TimeSeriesDatafile(Datafile):
    def load_data(self, formatter):
        super(TimeSeriesDatafile, self).load_data(formatter)


PipelineDatafile = TimeSeriesDatafile
