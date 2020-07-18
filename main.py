from shalconv.funcphys import *
from shalconv.serialization import read_data, compare_data, OUT_VARS


if __name__ == "__main__":
    
    ser_count_max = 10
    num_tiles = 6
    
    # Initialization
    gpvs()
    
    for tile in range(0, num_tiles):
        
        # Read serialized data
        input_data = read_data(i, "in")
        
        numpy_dict_to_gt4py_dict(input_data, BACKEND)
        
        # Main algorithm
        samfshalcnv(input_data)
