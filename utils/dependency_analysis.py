import re
def find_occurances_in_lines(var_list, filename):
    line_dict = {key:[] for key in var_list}
    re_dict = {key:re.compile("\W"+key+"\W") for key in var_list}
    with open(filename, "r") as f:
        line_num = 0
        for line in f:
            line_num += 1
            for var in var_list:
                if re_dict[var].search(line):
                #if var in line:
                    line_dict[var].append(line_num)
    return line_dict

def filter_occurances_in_linerange(line_dict, linerange):
    start, end = linerange
    new_line_dict = {}
    for var in line_dict.keys():
        lines = line_dict[var]
        new_line_dict[var] = [line for line in lines if line in range(start, end + 1)]
    return new_line_dict