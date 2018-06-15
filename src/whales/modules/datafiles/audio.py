from whales.modules.datafiles.time_series import TimeSeriesDatafile


class AudioDatafile(TimeSeriesDatafile):
    description = "Audio data files"

    @property
    def duration(self):
        return self.metadata["num_frames"] / self.metadata["frame_rate"]


PipelineDatafile = AudioDatafile
