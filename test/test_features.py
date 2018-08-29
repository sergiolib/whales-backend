import sys, os

import pytest

from whales.modules.features_extractors.feature_extraction import FeatureExtraction

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
from whales.modules.features_extractors.kurtosis import PipelineMethod as Kurtosis
from whales.modules.features_extractors.energy import PipelineMethod as Energy
from whales.modules.features_extractors.spectrogram import PipelineMethod as SpectralFrames


def test_spectral_frames():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                              formatter=AIFFormatter())
    df.data -= df.data.mean()
    f = SpectralFrames()
    f.parameters["sampling_rate"] = df.sampling_rate
    f.private_parameters["data"] = df
    t = f.transform()
    assert t.data.values.shape[0] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_mfcc():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                              formatter=AIFFormatter())
    df.data -= df.data.mean()
    f = MFCC()
    f.parameters["sampling_rate"] = df.sampling_rate
    f.parameters["n_components"] = 25
    f.private_parameters["data"] = df
    t = f.transform()
    assert t.data.values.shape[1] == 25
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_identity():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                                   formatter=AIFFormatter())
    f = Identity()
    f.private_parameters["data"] = df
    t = f.transform()
    np.testing.assert_allclose(df.data.values.ravel(), t.data.values.ravel())
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_zero_crossing_rate():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                                   formatter=AIFFormatter())
    df.data -= df.data.mean()
    f = ZeroCrossingRate()
    f.private_parameters["data"] = df
    t = f.transform()
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_min():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                                   formatter=AIFFormatter())
    df.data -= df.data.mean()
    f = Min()
    f.private_parameters["data"] = df
    t = f.transform()
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_range():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                                   formatter=AIFFormatter())
    df.data -= df.data.mean()
    f = Range()
    f.private_parameters["data"] = df
    t = f.transform()
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_skewness():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                                   formatter=AIFFormatter())
    df.data -= df.data.mean()
    f = Skewness()
    f.private_parameters["data"] = df
    t = f.transform()
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_kurtosis():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                              formatter=AIFFormatter())
    df.data -= df.data.mean()
    f = Kurtosis()
    f.private_parameters["data"] = df
    t = f.transform()
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_energy():
    file_name = get_file_name()
    df = AudioDataFile().load(file_name,
                                   formatter=AIFFormatter())
    df.data -= df.data.mean()
    f = Energy()
    f.private_parameters["data"] = df
    t = f.transform()
    assert t.data.values.shape[1] == 1
    assert f.description != ""
    assert t.data.values.ndim == 2


def test_unfitted():
    f = FeatureExtraction()
    f.needs_fitting = True
    with pytest.raises(RuntimeError):
        f.transform()


def test_unimplemented():
    f = FeatureExtraction()
    with pytest.raises(NotImplementedError):
        f.method_fit()
    with pytest.raises(NotImplementedError):
        f.method_transform()


def test_incorrect_data_type():
    f = FeatureExtraction()
    f.needs_fitting = True
    f.private_parameters["data"] = np.random.rand(10, 10)
    with pytest.raises(AttributeError):
        f.fit()
