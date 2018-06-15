from urllib.request import urlretrieve


def get_filename(desired_path=None):
    """Download and place a working AIFF sample file to perform the tests"""
    desired_path = desired_path or "ballena_bw_ruido_002_PU145_20120330_121500.aif"
    url = "https://www.cec.uchile.cl/~sliberman/ballena_bw_ruido_002_PU145_20120330_121500.aif"
    urlretrieve(url, desired_path)
    return desired_path
