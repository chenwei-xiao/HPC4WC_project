from shalconv.serialization import read_random_input, read_data, numpy_dict_to_gt4py_dict, scalar_vars
from shalconv.samfshalcnv import samfshalcnv_func
from shalconv import DTYPE_INT, BACKEND
from time import time
import numpy as np

def run_model(ncolumns, nrun = 10):
    ser_count_max = 19
    num_tiles = 6
    input_0 = read_data(0, True)
    ix = input_0["ix"]
    length = DTYPE_INT(ncolumns)
    times = np.zeros(nrun)
    for i in range(nrun):
        data = read_random_input(length, ix, num_tiles, ser_count_max)
        for key in scalar_vars:
            data[key] = input_0[key]
        data["ix"] = length
        data["im"] = length
        data = numpy_dict_to_gt4py_dict(data)
        if i == 0: samfshalcnv_func(data)
        tic = time()
        samfshalcnv_func(data)
        toc = time()
        times[i] = toc - tic
    return times

if __name__ == "__main__":
    lengths = [32, 128, 512, 2048, 8192, 32768, 131072, 524288]
    nrun = 10
    time_mat = np.zeros((nrun, len(lengths)))
    print(f"Benchmarking samfshalcnv with backend: {BACKEND}")
    for i in range(len(lengths)):
        length = lengths[i]
        times = run_model(length, nrun)
        time_mat[:,i] = times
        print(f"ix = {length}, Run time: Avg {times.mean():.3f}, Std {np.std(times):.3e} seconds")
    np.savetxt(f"times-{BACKEND}.csv", time_mat, delimiter=",")