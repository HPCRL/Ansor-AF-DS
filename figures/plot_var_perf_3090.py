import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D

# Function to read the aggregated data
def read_aggregated_data(filepath):
    return pd.read_csv(filepath)

# Corrected function to add bars for a single subplot
def add_subplot_bars(ax, testcase_map, index, bar_width, total_bars=3):
    # Calculate the total width of all bars together
    total_width = total_bars * bar_width
    
    # Calculate offset to center the group of bars
    offset = (total_width - bar_width) / 2
    
    for i, (testcase_key, data) in enumerate(testcase_map.items()):
        ansor_data, our_data = data
        
        # Calculate each bar's position
        ansor_pos = index[i] - offset
        # our_1min_pos = index[i] - offset + bar_width
        
        # our_2min_pos = index[i] - offset + 2 * bar_width
        # our_best_pos = index[i] - offset + 3 * bar_width
        
        our_2min_pos = index[i] - offset + 1 * bar_width
        our_best_pos = index[i] - offset + 2 * bar_width

        ansor_best_error = (ansor_data['best_max_gflops'] - ansor_data['best_min_gflops']) / 2
        our_best_error = (our_data['best_max_gflops'] - our_data['best_min_gflops']) / 2
        our_1_error = (our_data['1_max_gflops'] - our_data['1_min_gflops']) / 2
        our_2_error = (our_data['2_max_gflops'] - our_data['2_min_gflops']) / 2
        
        # Plot ansor best bars
        ax.bar(ansor_pos, ansor_data['best_mean_gflops'], bar_width,
            bottom=0, yerr=ansor_best_error, 
            label='Ansor 1000 trials' if i == 0 else '', color='#44AA99', edgecolor='grey', capsize=5)

        # Plot our 2min bars
        ax.bar(our_2min_pos, our_data['2_mean_gflops'], bar_width,
            bottom=0, yerr=our_2_error, 
            label='Ansor-AF-DS(2 min)' if i == 0 else '', color='#CC6677', edgecolor='grey', capsize=5)
        
        # Plot our best bars
        ax.bar(our_best_pos, our_data['best_mean_gflops'], bar_width,
            bottom=0, yerr=our_best_error, 
            label='Ansor-AF-DS 1000 trials' if i == 0 else '', color='#FFC20A', edgecolor='grey', capsize=5)
        
        # # Plot our 1min bars
        # ax.bar(our_1min_pos, our_data['1_mean_gflops'], bar_width,
        #     bottom=0, yerr=our_1_error,
        #     label='Ansor-AF-DS(1 min)' if i == 0 else '', color='#CC6677', edgecolor='grey', capsize=5)
        
        
        # star(*): min gflops, cross(x): max gflops
        # Plot min/max gflops for each
        
        # # Plot min/max for Ansor best
        # ax.plot(ansor_pos, ansor_data['best_max_gflops'], 'x', color='#44AA99', markersize=12, markeredgewidth=2, zorder=10)
        # ax.plot(ansor_pos, ansor_data['best_min_gflops'], '*', color='blue', markersize=12, markeredgewidth=2, zorder=10)
        
        
        # # Plot min/max for our best
        # ax.plot(our_best_pos, our_data['best_max_gflops'], 'x', color='orange', markersize=12, markeredgewidth=2, zorder=10)
        # ax.plot(our_best_pos, our_data['best_min_gflops'], '*', color='blue', markersize=12, markeredgewidth=2, zorder=10)
        
        # # Plot min/max for our 1min
        # ax.plot(our_1min_pos, our_data['1_max_gflops'], 'x', color='purple', markersize=12, markeredgewidth=2, zorder=10)
        # ax.plot(our_1min_pos, our_data['1_min_gflops'], '*', color='blue', markersize=12, markeredgewidth=2, zorder=10)
        
        # # Plot min/max for our 2min
        # ax.plot(our_2min_pos, our_data['2_max_gflops'], 'x', color='pink', markersize=12, markeredgewidth=2, zorder=10)
        # ax.plot(our_2min_pos, our_data['2_min_gflops'], '*', color='blue', markersize=12, markeredgewidth=2, zorder=10)

network_mapping = {
    'mm': 'M',
    'resnet': 'R',
    'yolo': 'Y'
}


# Function to plot all test cases from all networks in subplots
def plot_all_testcases_in_subplots(data_folders, networks):
    # Maximum number of test cases per subplot
    max_testcases_per_subplot = 17
    bar_width = 0.25
    # Build the testcase map
    testcase_map = {}
    print(f"processing data_folders = {data_folders}")
    for data_folder in data_folders:
        for network in networks:
            ansor_data = read_aggregated_data(f'{data_folder}/ansor/{network}_aggregated_data.csv')
            our_data = read_aggregated_data(f'{data_folder}/our/{network}_aggregated_data.csv')

            for testcase in ansor_data['testcase_']:
                key = f"{network}{testcase}"
                # if key in yolo_mapping:
                #     # change to 
                #     key = yolo_mapping[key]
                # else:
                #     key = key.replace(network, network_mapping[network])
                key = key.replace(network, network_mapping[network])
                print(key)
                testcase_map[key] = (
                    ansor_data[ansor_data['testcase_'] == testcase].iloc[0],
                    our_data[our_data['testcase_'] == testcase].iloc[0],
                )
    
    # Determine how many subplots we need
    num_subplots = int(np.ceil(len(testcase_map) / max_testcases_per_subplot))
    
    # Specify the relative heights of each subplot
    height_ratios = [5, 5]  # Adjust the values based on your preference

    # Create subplots with specified heights
    fig, axs = plt.subplots(num_subplots, 1, figsize=(18, sum(height_ratios)), gridspec_kw={'height_ratios': height_ratios})

    # # Custom legend entries
    # green_cross = Line2D([0], [0], linestyle="none", marker='x', color='#44AA99', markersize=10, markeredgewidth=2, label='max gflops')
    # red_star = Line2D([0], [0], linestyle="none", marker='*', color='#CC6677', markersize=10, markeredgewidth=2, label='min gflops')
    
    if num_subplots == 1:
        axs = [axs]  # Make sure axs is iterable even when there is only one subplot
    
    # Add bars to each subplot
    for i in range(num_subplots):
        start_idx = i * max_testcases_per_subplot
        end_idx = min(start_idx + max_testcases_per_subplot, len(testcase_map))
        subplot_testcase_map = dict(list(testcase_map.items())[start_idx:end_idx])
        index = np.arange(len(subplot_testcase_map))
        
        add_subplot_bars(axs[i], subplot_testcase_map, index, bar_width)
        if i == 0:
            middle_bar_index = 5.5
            axs[i].axvline(middle_bar_index, color='grey', linestyle='--')

        axs[i].set_xticks(index)
        print(subplot_testcase_map.keys())
        axs[i].set_xticklabels(subplot_testcase_map.keys(), rotation=0, fontsize=20)
        axs[i].tick_params(axis='y', labelsize=20)
        axs[i].set_ylabel('GFLOPS', fontsize=20)
        # set front size of ylabel
        axs[i].yaxis.label.set_size(20)
        # if i == 0:  # Add legend only to the first subplot
        #     axs[i].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=20, prop={'size': 20}, framealpha=1, facecolor='white', edgecolor='black', ncol=3)
        
        # if i == 0:  # Add legend only to the first subplot
        #     handles, labels = axs[i].get_legend_handles_labels()
        #     print(f"handles = {handles}")
        #     handles.append(green_cross)
        #     handles.append(red_star)
        #     axs[i].legend(handles=handles, loc='best', fontsize=18, prop={'size': 18}, framealpha=1, ncol=4)

    for ax in axs:
        if ax == axs[0]:
            ax.set_ylim(0, 17500)
        else:
            ax.set_ylim(0, 11000)
        
        
        # get yticks
        yticks = ax.get_yticks()
        # convert to TFLOPS
        new_yticks = yticks / 1000

        # set new_yticks
        ax.yaxis.set_major_locator(FixedLocator(yticks)) 
        ax.set_yticklabels(['{:.0f}'.format(ytick) for ytick in new_yticks])

        # set ylabel
        ax.set_ylabel('TFLOPS', fontsize=20)
        
    for ax in axs:
        ax.text(0.89, 0.70, 'RTX 3090', transform=ax.transAxes, fontsize=20, verticalalignment='top', zorder=10)

    # fig.suptitle('GFLOPS Performance Comparison', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    machine = data_folders[0].split('/')[-1].split('csv')[0]
    plt.savefig(f'{machine}_perf_var.pdf')


# # Define data folders and networks
# data_folders = ['/home/chendi/githdd/plot/data/4090csv', '/home/chendi/githdd/plot/data/3090csv']


# data_folders = ['/home/chendi/githdd/plot/data/4090csv']
data_folders = ['3090csv_var']
networks = ['mm', 'yolo', 'resnet']

for dir in data_folders:
    data_folder = []
    data_folder.append(dir)
    # Plot all test cases
    plot_all_testcases_in_subplots(data_folder, networks)
