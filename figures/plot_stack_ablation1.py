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
    offset = (total_width - bar_width) / 3
    
    for i, (testcase_key, data) in enumerate(testcase_map.items()):
        ansor_data, our_fea_GA_data = data
        
        # Calculate each bar's position
        ansor_pos = index[i] - offset
        our_fea_GA_pos = index[i] - offset + bar_width

        # Plot our_fea_GA bars
        # Plot our_fea_GA bars
        # print(f"our_fea_GA_data['1_mean_gflops'] = {our_fea_GA_data['1_mean_gflops']}")
        # print(f"our_fea_GA_data['2_mean_gflops'] = {our_fea_GA_data['2_mean_gflops']}")
        # Check if '1_mean_gflops' is NaN and replace it with 0
        
        
        # ansor_1_error = (ansor_data['1_max_gflops'] - ansor_data['1_mean_gflops']) / ansor_data['1_mean_gflops']**2
        # ansor_2_error = (ansor_data['2_max_gflops'] - ansor_data['2_mean_gflops']) / ansor_data['2_mean_gflops']**2
        
        # our_1_error = (our_fea_GA_data['1_max_gflops'] - our_fea_GA_data['1_mean_gflops']) / our_fea_GA_data['1_mean_gflops']**2
        # our_2_error = (our_fea_GA_data['2_max_gflops'] - our_fea_GA_data['2_mean_gflops']) / our_fea_GA_data['2_mean_gflops']**2
        
        # # Plot ansor bars
        # ax.bar(ansor_pos, ansor_data['1_mean_gflops'], bar_width, yerr=ansor_1_error, label='Ansor (1 min)' if i == 0 else '', color='lightgreen', edgecolor='grey', capsize=5)
        # if np.isnan(ansor_data['1_mean_gflops']):
        #     ax.bar(ansor_pos, ansor_data['2_mean_gflops'], bar_width, yerr=ansor_2_error, label='Ansor (2 mins)' if i == 0 else '', color='green', edgecolor='grey', capsize=5)
        # else:
        #     ax.bar(ansor_pos, ansor_data['2_mean_gflops'] - ansor_data['1_mean_gflops'], bar_width, bottom=ansor_data['1_mean_gflops'], label='Ansor (2 mins)' if i == 0 else '', color='green', edgecolor='grey', capsize=5)
        
        ax.bar(ansor_pos, ansor_data['2_mean_gflops'], bar_width, label='Ansor (2 mins)' if i == 0 else '', color='#117733', edgecolor='grey', capsize=5)

        # # Plot our_fea_GA bars
        # ax.bar(our_fea_GA_pos, our_fea_GA_data['1_mean_gflops'], bar_width, label='Ansor AF (1 min)' if i == 0 else '', color='yellow', edgecolor='grey', capsize=5)
        # # Check if '1_mean_gflops' is NaN and replace it with 0
        # if np.isnan(our_fea_GA_data['1_mean_gflops']):
        #     ax.bar(our_fea_GA_pos, our_fea_GA_data['2_mean_gflops'], bar_width, label='Ansor AF (2 mins)' if i == 0 else '', color='orange', edgecolor='grey', capsize=5)
        # else:
        #     ax.bar(our_fea_GA_pos, our_fea_GA_data['2_mean_gflops'] - our_fea_GA_data['1_mean_gflops'], bar_width, bottom=our_fea_GA_data['1_mean_gflops'], label='Ansor AF (2 mins)' if i == 0 else '', color='orange', edgecolor='grey', capsize=5)
            
            
        ax.bar(our_fea_GA_pos, our_fea_GA_data['2_mean_gflops'], bar_width, label='Ansor-AF (2 mins)' if i == 0 else '', color='#FDB5B5', edgecolor='grey', capsize=5)

        # Plot stars for best_mean_gflops (ansor)
        ansor_best_mean = ansor_data['best_mean_gflops']
        ax.plot(ansor_pos, ansor_best_mean, 'x', color='#117733', markersize=8, markeredgewidth=2)
        
        # Plot stars for best_mean_gflops (our_fea_GA)
        our_fea_GA_best_mean = our_fea_GA_data['best_mean_gflops']
        ax.plot(our_fea_GA_pos, our_fea_GA_best_mean, '*', color='#FDB5B5', markersize=10, markeredgewidth=2)

network_mapping = {
    'mm': 'M',
    'resnet': 'R',
    'yolo': 'Y'
}

# Function to plot all test cases from all networks in subplots
def plot_all_testcases_in_subplots(data_folders, networks):
    # Maximum number of test cases per subplot
    max_testcases_per_subplot = 35
    bar_width = 0.35
    # Build the testcase map
    testcase_map = {}
    for data_folder in data_folders:
        for network in networks:
            ansor_data = read_aggregated_data(f'{data_folder}/ansor/{network}_aggregated_data.csv')
            our_fea_GA_data = read_aggregated_data(f'{data_folder}/our_fea_GA/{network}_aggregated_data.csv')

            for testcase in ansor_data['testcase_']:
                key = f"{network}{testcase}"
                key = key.replace(network, network_mapping[network])
                print(key)
                matching_rows = our_fea_GA_data[our_fea_GA_data['testcase_'] == testcase]
                if not matching_rows.empty:
                    testcase_map[key] = (
                        ansor_data[ansor_data['testcase_'] == testcase].iloc[0],
                        matching_rows.iloc[0],
                    )
                else:
                    print(f"No matching rows for {testcase} in {network}")
    # Determine how many subplots we need
    num_subplots = int(np.ceil(len(testcase_map) / max_testcases_per_subplot))
    
    # Create subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(15, 5))
    if num_subplots == 1:
        axs = [axs]  # Make sure axs is iterable even when there is only one subplot
    
    green_cross = Line2D([0], [0], linestyle="none", marker='x', color='#117733', markersize=10, markeredgewidth=2, label='Ansor 1000 trials')
    pink_star = Line2D([0], [0], linestyle="none", marker='*', color='#FDB5B5', markersize=10, markeredgewidth=2, label='Ansor-AF 1000 trials')

    # Add bars to each subplot
    for i in range(num_subplots):
        start_idx = i * max_testcases_per_subplot
        end_idx = min(start_idx + max_testcases_per_subplot, len(testcase_map))
        subplot_testcase_map = dict(list(testcase_map.items())[start_idx:end_idx])
        index = np.arange(len(subplot_testcase_map))
        
        add_subplot_bars(axs[i], subplot_testcase_map, index, bar_width)
        axs[i].set_xticks(index)
        axs[i].set_xticklabels(subplot_testcase_map.keys(), rotation=0, fontsize=15)
        axs[i].tick_params(axis='y', labelsize=15)
        # set front size of ylabel
        axs[i].yaxis.label.set_size(20)
        # divide numbers to change gflops to tflops
        # axs[i].set_ylabel('GFLOPS', fontsize=20)
        if i == 0:  # Add legend only to the first subplot
            handles, labels = axs[i].get_legend_handles_labels()
            # Add custom legend entries by appending them to the handles list
            handles.extend([green_cross, pink_star])
            axs[i].legend(handles=handles, loc='best', fontsize=16, prop={'size': 15}, framealpha=1, facecolor='white', edgecolor='black')
    
    for ax in axs:
        # get yticks
        yticks = ax.get_yticks()

        # convert to TFLOPS
        new_yticks = yticks / 1000

        # set new_yticks
        ax.yaxis.set_major_locator(FixedLocator(yticks)) 
        ax.set_yticklabels(['{:.0f}'.format(ytick) for ytick in new_yticks])

        # set ylabel
        ax.set_ylabel('Performance (TFlops)', fontsize=20)
        
    # fig.suptitle('GFLOPS Performance Comparison', fontsize=18)
    for ax in axs:
        ax.set_xlim(left=index[0] - 0.6, right=index[-1] + 0.4)
        
    plt.tight_layout()
    plt.savefig('ablation1.pdf')

### 4090
### 4090


# Define data folders and networks
data_folders = ['raw_ablation1']
networks = ['mm', 'yolo', 'resnet']

# Plot all test cases
plot_all_testcases_in_subplots(data_folders, networks)
