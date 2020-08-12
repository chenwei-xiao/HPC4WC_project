from shalconv.serialization import *
from shalconv.samfshalcnv import samfshalcnv_func
from shalconv import *

if __name__ == "__main__":
    
    ser_count_max = 19
    num_tiles = 6

    for tile in range(0, num_tiles):
        for ser_count in range(ser_count_max):
            # Read serialized data
            print(f"Comparing tile {tile}, ser_count {ser_count}")
            input_data = read_data(tile, True, path=DATAPATH, ser_count=ser_count)
            output_data = read_data(tile, False, path=DATAPATH, ser_count=ser_count)

            gt4py_dict = numpy_dict_to_gt4py_dict(input_data)

            # Main algorithm
            kcnv, kbot, ktop, q1, t1, u1, v1, rn, cnvw, cnvc, ud_mf, dt_mf, qtr = samfshalcnv_func(gt4py_dict)
            exp_data = {"kcnv": kcnv[0,:,0].view(np.ndarray),  "kbot": kbot[0,:,0].view(np.ndarray),
                        "ktop": ktop[0,:,0].view(np.ndarray),  "q1":   q1[0,:,:].view(np.ndarray),
                        "t1":   t1[0,:,:].view(np.ndarray),    "u1":   u1[0,:,:].view(np.ndarray),
                        "v1":   v1[0,:,:].view(np.ndarray),    "rn":   rn[0,:,0].view(np.ndarray),
                        "cnvw": cnvw[0,:,:].view(np.ndarray),  "cnvc": cnvc[0,:,:].view(np.ndarray),
                        "ud_mf":ud_mf[0,:,:].view(np.ndarray), "dt_mf":dt_mf[0,:,:].view(np.ndarray),
                        "qtr":  qtr[:,:,:].view(np.ndarray)}
            ref_data = {"kcnv":  output_data["kcnv"],  "kbot":  output_data["kbot"],
                        "ktop":  output_data["ktop"],  "q1":    output_data["q1"],
                        "t1":    output_data["t1"],    "u1":    output_data["u1"],
                        "v1":    output_data["v1"],    "rn":    output_data["rn"],
                        "cnvw":  output_data["cnvw"],  "cnvc":  output_data["cnvc"],
                        "ud_mf": output_data["ud_mf"], "dt_mf": output_data["dt_mf"],
                        "qtr":   output_data["qtr"]}
            compare_data(exp_data, ref_data)
