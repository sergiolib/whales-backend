import numpy as np
from python_speech_features import mfcc
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class MFCC(FeatureExtraction):
    "Mel Frequency Cepstral Coefficients"
    def __init__(self, logger=None):
        super(MFCC, self).__init__(logger)
        self.needs_fitting = False
        self.parameters = {
            # TODO: add relevant parameters
        }

    def method_transform(self, data):
        out = []
        for d in data:
            d = d.reshape(-1, 1)
            f = mfcc(d, **self.parameters)
            out.append(f.ravel())
        res = np.vstack(out)
        return res


PipelineMethod = MFCC
