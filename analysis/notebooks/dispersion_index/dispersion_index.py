from typing import List, Optional, IO


SAMPLING_RES = 60  # Sampling resolution, T, (e.g. 60ms)
CONVERGENCE_TOL = 0.20  # Convergence tolerance, e.g. 0.20
LOG_FILE_PATH = "./loadgen.log"


def get_dispersion_index(
    sampling_res: int,
    convergence_tol: float,
    log_file_path: Optional[str] = None,
    log_file_obj: Optional[IO] = None,
) -> Optional[float]:
    """
    Get Dispersion Index from request logs.

    Usage: python3 dispersion_index.py \
        --sampling-res=60 \
        --convergence_tol=0.05 \
        --log-file-path="./loadgen.log"

    Args:
        sampling_res: (int) the sampling resolution, in milliseconds (e.g. 60ms).
            Corresponds to T in the algorithm described in Fig 2. of [1].
        convergence_tol: (float) convergence tolerance, e.g. 0.20.
            Corresponds to tol in the algorithm described in Fig 2. of [1].
        log_file_path: (str) absolute or relative file path to the log file
            containing the request timestamps per line. The implementation assumes
            timestamps are in the format parsable by the regex pattern used.
        log_file_obj: (file object) An open file object of the loadgen log file.
            The file is opened by the caller. Typically, a TextIO type.

        Either log_file_path or log_file_obj must be provided.

    Returns:
        The index of dispersion if convergence can be achieved.

    Notes:
        1. Convergence is not guaranteed if the sampling resolution is too big.
        2. Highly affected by choice of sampling resolution. (I am not exactly sure why)
        3. Tested using Python 3.8.2.

    References:
        [1]. Burstiness in Multi-tier Applications: Symptoms, Causes, and New Models
            (https://link.springer.com/chapter/10.1007/978-3-540-89856-6_14)
    """
    import re
    from datetime import datetime, timedelta
    from statistics import mean, variance
    from bisect import bisect_left, bisect_right

    timestamp_regex = (
        "[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1]) "
        "(2[0-3]|[01][0-9]):[0-5][0-9]:[0-5][0-9].[0-9]{3}"
    )

    timestamps: List[datetime] = []

    assert log_file_path or log_file_obj, "Missing log path or log file object"

    # Read the log file & extract the timestamp in each line.
    file = log_file_obj if log_file_obj else open(file=log_file_path, mode="r")
    for line in file:
        ts_str: str = re.search(
            pattern=timestamp_regex,
            # Binary I/O might need to be properly decoded.
            string=line.decode("utf-8") if log_file_obj else line,
        ).group()
        ts: datetime = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        timestamps.append(ts)
    file.close()

    # Sort the timestamps.
    timestamps.sort()

    window: int = sampling_res
    last_window_index: Optional[float] = None

    print("Estimating Index of Dispersion for request timestamps...")

    # Loop over increasing time window, t.
    while True:
        sample: List[int] = []

        # Gather sample for current window.
        start_ts: datetime = timestamps[0]
        while True:
            end_ts: datetime = start_ts + timedelta(milliseconds=window)
            if end_ts > timestamps[-1]:
                # Might not be a busy period for entire window.
                break

            start_inx: int = bisect_left(a=timestamps, x=start_ts)
            end_inx: int = bisect_right(a=timestamps, x=end_ts, lo=start_inx)
            requests_in_window: int = end_inx - start_inx
            sample.append(requests_in_window)
            start_ts = end_ts

        if len(sample) < 100:
            print(f"Sample for time window = {window}ms is too small.")
            print(f"Last index = {last_window_index}")
            print(f"Exiting without converging.")
            break

        # Dispersion Index for the current window.
        curr_window_index: float = variance(sample) / mean(sample)

        if last_window_index is not None:
            try:
                error: float = abs(1 - (curr_window_index / last_window_index))
                print(f"Relative error = {error} for window = {window}ms")
            except ZeroDivisionError:
                # Handle uniform distribution, when dispersion is 0.
                error = abs(last_window_index - curr_window_index)

            if error <= convergence_tol:
                print(f"Converged to Dispersion Index = {curr_window_index}")
                return curr_window_index

        last_window_index = curr_window_index
        window += sampling_res


if __name__ == "__main__":
    import fire as fire

    fire.Fire(get_dispersion_index)
