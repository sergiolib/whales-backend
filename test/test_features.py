import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np
from whales.modules.data_files.audio import AudioDataFile
from whales.modules.formatters.aif import AIFFormatter
from whales.utilities.testing import get_file_name
from whales.modules.features_extractors.mfcc import PipelineMethod as MFCC
from whales.modules.features_extractors.identity import PipelineMethod as Identity
from whales.modules.features_extractors.zero_crossing_rate import PipelineMethod as ZeroCrossingRate
from whales.modules.features_extractors.min import PipelineMethod as Min
from whales.modules.features_extractors.range import PipelineMethod as Range
from whales.modules.features_extractors.skewness import PipelineMethod as Skewness
from whales.modules.features_extractors.energy import PipelineMethod as Energy
from whales.modules.features_extractors.spectral_frames import PipelineMethod as SpectralFrames


def generate_data(n, d):
    return np.random.rand(n, d)


# def test_spectral_frames():
#     file_name = get_file_name()
#     df = AudioDataFile().load_data(file_name,
#                                    formatter=AIFFormatter())
#     data = df.data.values.ravel()
#     data = data / abs(data).max()
#     parameters = {
#         "win": 4096,
#         "step": 2048
#     }
#     f = SpectralFrames()
#     f.parameters = parameters
#     t = f.transform(data=data)
#     assert t.data.values.ndim == 2
#     assert f.description != ""


def test_mfcc():
    file_name = get_file_name()
    df = AudioDataFile().load_data(file_name,
                                   formatter=AIFFormatter())
    data = df.data.drop("labels", axis="columns").values
    data = data / abs(data).max()
    parameters = {
        "win": 4096,
        "step": 2048,
        "to_db": True
    }
    f = MFCC()
    f.parameters = parameters
    t = f.transform(data=data)
    #assert t.shape[0] == data.shape[0]
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_identity():
    data = generate_data(10000, 2500)
    f = Identity()
    t = f.transform(data=data)
    np.testing.assert_allclose(data, t.data.values)
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_zero_crossing_rate():
    data = generate_data(10000, 2500) - 0.5
    f = ZeroCrossingRate()
    t = f.transform(data=data)
    assert t.data.values.shape[0] == data.shape[0]
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_min():
    data = generate_data(10000, 2500) - 0.5
    f = Min()
    t = f.transform(data=data)
    assert t.data.values.shape[0] == data.shape[0]
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_range():
    data = generate_data(10000, 2500) - 0.5
    f = Range()
    t = f.transform(data=data)
    assert t.data.values.shape[0] == data.shape[0]
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_skewness():
    data = generate_data(10000, 2500) - 0.5
    f = Skewness()
    t = f.transform(data=data)
    assert t.data.values.shape[0] == data.shape[0]
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_energy():
    data = generate_data(10000, 2500) - 0.5
    f = Energy()
    t = f.transform(data=data)
    assert t.data.values.shape[0] == data.shape[0]
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2
