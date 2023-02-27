"""
Read the results text file and return a dictionary of all experiments
"""


def read_file_to_list_of_dicts(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        dict_strings = content.split("\n}")
        dicts = []
        for dict_string in dict_strings:
            if dict_string:
                kv_pairs = dict_string.strip().split("\n")
                current_dict = {}
                for kv in kv_pairs:
                    if ":" in kv:
                        key = kv.split(":")[0].strip().strip('"')
                        value = kv.split(":")[1].strip().strip('"').strip(',')
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                        current_dict[key] = value
                dicts.append(current_dict)
        return dicts


dicts = read_file_to_list_of_dicts("results.txt")
