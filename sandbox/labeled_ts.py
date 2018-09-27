import logging

from whales.modules.pipelines.predict_whale_detector import PredictWhaleDetector

logging.basicConfig(level=logging.DEBUG)
from whales.modules.pipelines.train_whale_detector import TrainWhaleDetector

params = {
    "input_data": [{
        "file_name": "/home/sliberman/Dropbox/Detector ballena azul/supervised_version/database/validation-ballenas/ballenas-S06_PU145_20120514_143000.aif",
        "data_file": "audio",
        "formatter": "aif"
    }],
    "input_labels": [{
        "labels_file": "/home/sliberman/Dropbox/Detector ballena azul/supervised_version/database/etiquetas/csv/Chile01_002K_S06_PU145_20120514_143000-Labels.csv",
        "labels_formatter": "csv"
    }],
    "logs_directory": ".",
    "models_directory": ".",
    "results_directory": ".",
    "machine_learning": {
        "type": "supervised",
        "method": "svm"
    },
    "features_extractors": [
        {
            "method": "zero_crossing_rate"
        }
    ],
    "pre_processing": [
        {
            "method": "sliding_windows",
            "parameters": {
                "window_size": "1000ms",
                "overlap": 0.3
            }
        }
    ],
    "performance_indicators": [
        {
            "method": "accuracy"
        },
        {
            "method": "labeled_time_series"
        }
    ]
}

p = TrainWhaleDetector()
p.load_parameters(params)
p.initialize()
p.start()

params.update({
    "performance_indicators": [
        {
            "method": "labeled_spectrogram"
        }
    ],
    "input_data": [{
            "file_name": "/home/sliberman/Dropbox/Detector ballena azul/supervised_version/database/test-ballenas/ballenas-009_PU145_20120214_211500.aif",
            "data_file": "audio",
            "formatter": "aif"
        }],
    "input_labels": [{
        "labels_file": "/home/sliberman/Dropbox/Detector ballena azul/supervised_version/database/etiquetas/csv/ballenas-bw_009_PU145_20120214_211500-Labels.csv",
        "labels_formatter": "csv"
    }]
})

p = PredictWhaleDetector()
p.load_parameters(params)
p.initialize()
p.start()