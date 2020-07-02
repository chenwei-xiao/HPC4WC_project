from shalconv.serialization import read_data, compare_data, OUT_VARS

def test_read_inputvsinput():
    for i in range(6):
        data = read_data(i,"in")
        compare_data(data, data)
        
def test_read_inputvsoutput(): #should FAIL!
    for i in range(6):
        in_data = read_data(i,"in")
        out_data = read_data(i, "out")
        in_data_filtered = {in_data[k] for k in OUT_VARS}
        compare_data(in_data_filtered, out_data)