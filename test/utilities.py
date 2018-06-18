from os.path import basename, join

from urllib.request import urlretrieve


def get_filename(desired_path=None):
    """Download and place a working AIFF sample file to perform the tests"""
    desired_path = desired_path or "ballena_bw_ruido_002_PU145_20120330_121500.aif"
    url = "https://www.cec.uchile.cl/~sliberman/ballena_bw_ruido_002_PU145_20120330_121500.aif"
    urlretrieve(url, desired_path)
    return desired_path


def get_5_filenames(desired_dir=None):
    """Download and place a working AIFF sample file to perform the tests"""
    filepaths = [
        "https://www.cec.uchile.cl/~sliberman/ballena_bw_ruido_002_PU145_20120330_121500.aif",
        "https://www.cec.uchile.cl/~sliberman/ballenas-bw_011_PU145_20120214_234500.aif",
        "https://www.cec.uchile.cl/~sliberman/ballenas-bw_012_PU145_20120215_223000.aif",
        "https://www.cec.uchile.cl/~sliberman/ballenas-bw_013_PU145_20120215_230000.aif",
        "https://www.cec.uchile.cl/~sliberman/ballenas-bw_015_PU145_20120330_120000.aif",
    ]
    res = []
    for url in filepaths:
        if desired_dir:
            desired_path = join(desired_dir, basename(url))
        else:
            desired_path = basename(url)
        urlretrieve(url, desired_path)
        res.append(desired_path)
    return res


def get_labeled(desired_path_data=None, desired_path_labels=None):
    """Download and place a working AIFF sample file to perform the tests"""
    desired_path_data = desired_path_data or "ballenas-bw_016_PU145_20120504_150000.aif"
    desired_path_labels = desired_path_labels or "ballenas-bw_016_PU145_20120504_150000-Labels.csv"
    url_data = "https://www.cec.uchile.cl/~sliberman/ballenas-bw_016_PU145_20120504_150000.aif"
    url_labels = "https://www.cec.uchile.cl/~sliberman/ballenas-bw_016_PU145_20120504_150000-Labels.csv"
    urlretrieve(url_data, desired_path_data)
    urlretrieve(url_labels, desired_path_labels)
    return desired_path_data, desired_path_labels
