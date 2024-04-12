import json
import csv
import os
import re
import matplotlib.pyplot as plt
import numpy as np


# get list of exe_time, each[r][0][0] from json file
def get_exe_time_list_from_json(json_path):
    exe_time_list = []
    with open(json_path, "r") as f:
        for line in f:
            if line:
                key = 'r'
                try:
                    data = json.loads(line)
                    time_value = data.get(key)
                    if time_value:
                        exe_time_list.append(time_value[0][0])
                    else:
                        exe_time_list.append(1e+10)
                        print(f"Warning: '{key}' key not found or time_value is None. for config {line}")
                except json.JSONDecodeError:
                    # Attempt to fix and decode again
                    line = re.sub(r', "s".*', '', line)+'}'
                    print(f"Error decoding JSON in line: {line}")
                    data = json.loads(line)
                    time_value = data.get(key)
                    print(f"after fix, time is {time_value[0][0]}")
                    if time_value:
                        exe_time_list.append(time_value[0][0])
                    else:
                        exe_time_list.append(1e+10)
            else:
                print("Error: file is empty.")
    return exe_time_list

# get list of exe_time, each[r][0][0] from json file
def get_timestamp_list_from_json(json_path):
    exe_time_list = []
    first_time_stamp = -1
    with open(json_path, "r") as f:
        for line in f:
            if line:
                key = 'r'
                try:
                    data = json.loads(line)
                    time_value = data.get(key)
                    if time_value:
                        if first_time_stamp == -1:
                            first_time_stamp = time_value[3]
                        exe_time_list.append(time_value[3]-first_time_stamp)
                    else:
                        exe_time_list.append(1e+10)
                        print(f"Warning: '{key}' key not found or time_value is None. for config {line}")
                except json.JSONDecodeError:
                    # Attempt to fix and decode again
                    line = re.sub(r', "s".*', '', line)+'}'
                    print(f"Error decoding JSON in line: {line}")
                    data = json.loads(line)
                    time_value = data.get(key)
                    print(f"after fix, time is {time_value[3]}"-first_time_stamp)
                    if time_value:
                        exe_time_list.append(time_value[3]-first_time_stamp)
                    else:
                        exe_time_list.append(1e+10)
            else:
                print("Error: file is empty.")
    return exe_time_list


ansorjson_files = [
'plot_scatter/res7_1.json',
]


ourjson_files = [
'plot_scatter/res7_2.json',
]

# get ansor data
ansor_exe_time = get_exe_time_list_from_json(ansorjson_files[0])

print(len(ansor_exe_time))

list_of_index = [ i for i in range(len(ansor_exe_time))]
print(len(list_of_index))
print(ansor_exe_time[:5])


# get our data
our_exe_time = get_exe_time_list_from_json(ourjson_files[0])

our_list_of_index = [ i for i in range(len(our_exe_time))]

# our_time_stamp = get_timestamp_list_from_json(ourjson_files[0])
# print(len(our_exe_time))
# print(len(our_time_stamp))

# assert len(our_exe_time) == len(our_time_stamp)
# print(our_exe_time[:5])
# print(our_time_stamp[:5])


res7_flops = 51380224

# Convert execution times to performance
ansor_performance = [res7_flops/time/1e+9 if time != 0 else 0 for time in ansor_exe_time]
our_performance = [res7_flops/time/1e+9 if time != 0 else 0 for time in our_exe_time]

# Plotting both performances on the same figure
plt.figure(figsize=(16, 9))
plt.scatter(list_of_index, ansor_performance, color='blue', s=21, label='Ansor run 1')
plt.scatter(our_list_of_index, our_performance, color='red', s=21, label='Ansor run 2')
plt.xlabel('Time', fontsize=21)
plt.ylabel('Performance in GFlops', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# (loc='upper right', bbox_to_anchor=(1, 1), fontsize=21, prop={'size': 20}, framealpha=1, facecolor='white', edgecolor='black', ncol=3)

legend = plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1), fontsize=21,
                    handlelength=0.5, handletextpad=1, borderaxespad=0.5)
legend.get_frame().set_linewidth(2)

# plt.tick_params(axis='y', labelsize=21)
# plt.tick_params(axis='x', which='major', labelsize=21)
plt.xticks(rotation=0)



plt.savefig('scatter_plot.pdf', bbox_inches='tight')


# header
# out_path = "_time_"+filename.replace(".json", ".csv")
# with open(out_path, 'w') as f:
#     f.write("exe_time\n")

# with open(out_path, 'a') as f:
#     for each in exe_time:
#         f.write(str(each) + '\n')