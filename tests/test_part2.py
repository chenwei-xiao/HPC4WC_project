import numpy as np
import numpy.f2py
import os
from shalconv.serialization import read_data

def set_init(initfunc, data_dict):
    im = data_dict["im"]
    ix = data_dict["ix"]
    km = data_dict["km"]
    clam = data_dict["clam"]
    c0s = data_dict["c0s"]
    c1 = data_dict["c1"]
    asolfac = data_dict["asolfac"]
    pgcon = data_dict["pgcon"]

def test_part2():
    data_dict = read_data(0, True, path="/data")

    os.system("f2py --f2cmap fortran/.f2py_f2cmap -c -m part2 fortran/part2.f90")
    import part2

if __name__ == "__main__":
    test_part2()