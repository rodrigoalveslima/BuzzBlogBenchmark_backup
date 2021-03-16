from datetime import datetime, timedelta
from math import isclose
from unittest import TestCase
import os

import numpy as np

from dispersion_index import get_dispersion_index

TEST_LOADGEN_FILE_PATH = "./test-loadgen.log"


class TestGetDispersionIndex(TestCase):
    def test_exponential_times(self):
        # Generate exponential distribution of request timestamps in loadgen.log.
        scale = 100.0
        ts = datetime.utcnow()
        with open(file=TEST_LOADGEN_FILE_PATH, mode="w") as file:
            for i in range(50_000):
                ts += timedelta(
                    milliseconds=np.random.exponential(scale=scale, size=None)
                )
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
                file.write(ts_str + "\n")

        index = get_dispersion_index(
            sampling_res=100,
            convergence_tol=0.025,
            log_file_path=TEST_LOADGEN_FILE_PATH,
        )
        assert isclose(a=index, b=1.0, rel_tol=1e-2)

    def test_uniform_times(self):
        # One request every 100 ms.
        ts = datetime.utcnow()
        with open(file=TEST_LOADGEN_FILE_PATH, mode="w") as file:
            for i in range(15_000):
                ts += timedelta(milliseconds=100)
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
                file.write(ts_str + "\n")

        for sampling_res in range(100, 1000, 100):
            assert (
                get_dispersion_index(
                    sampling_res=sampling_res,
                    convergence_tol=0.20,
                    log_file_path=TEST_LOADGEN_FILE_PATH,
                )
                == 0.0
            )

    def tearDown(self) -> None:
        os.remove(TEST_LOADGEN_FILE_PATH)
