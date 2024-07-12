import os
import numpy as np
import ast

count_fail = False
fail_configs = []

def read_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract MPD values
    mpd_line = lines[0].split(":")[1].strip()
    mpd_values = [float(value) for value in mpd_line[1:-1].split(",")]

    # Extract Chamfer values
    chamfer_line = lines[1].split(":")[1].strip()
    chamfer_values = [float(value) for value in chamfer_line[1:-1].split(",")]

    # Extract Actions
    actions_line = lines[2].split(":")[1].strip()
    actions = int(actions_line)

    # Extract Success
    success_line = lines[3].split(":")[1].strip()
    success = float(success_line)

    return mpd_values, chamfer_values, actions, success

def summarize_results(folder_path, target_configs):
    global count_fail, fail_configs
    # Summarize results for a given folder and target configurations
    mpd_list, chamfer_list, actions_list, success_list = [], [], [], []
    # Loop through all files in the folder

    for config in target_configs:
        # convert config to string and remove the empty space
        config = str(config).replace(" ", "")
        file_name = "{}.txt".format(config)
        if file_name not in os.listdir(folder_path):
            print("File {} not found".format(file_name))
            continue
        file_path = os.path.join(folder_path, file_name)
        mpd, chamfer, actions, success = read_results(file_path)
        mpd_list.append(mpd[-1])
        chamfer_list.append(chamfer[-1])
        actions_list.append(actions)
        success_list.append(success)
        if count_fail and not success:
            print("Fail: {}".format(config))

    # Calculate averages
    avg_mpd = [np.mean(mpd_list), np.std(mpd_list)]
    avg_chamfer = [np.mean(chamfer_list), np.std(chamfer_list)]
    avg_actions = [np.mean(actions_list), np.std(actions_list)]
    success = np.mean(success_list)

    print("Number Target Configurations Found in {}: {}".format(folder_path, len(mpd_list)))

    return avg_mpd, avg_chamfer, avg_actions, success

def summarize_all_results(base_path, method_folders, target_configs):
    global count_fail, fail_configs
    # Summarize results for all method folders and target configurations
    results = {}

    for method_folder in method_folders:
        # if method_folder == "P2P1_plus_Subgoal":
        #     count_fail = True
        # else:
        #     count_fail = False
        folder_path = os.path.join(base_path, method_folder)
        avg_mpd, avg_chamfer, avg_actions, success = summarize_results(folder_path, target_configs)
        results[method_folder] = {'avg_mpd': avg_mpd, 'avg_chamfer': avg_chamfer, 'avg_actions': avg_actions, 'success': success}

    return results

# Provide the base path, method folders, and target configurations
base_path = './data/'
method_folders = ['P2P1', 'P2P2', 'P2P1_plus_Subgoal', 'RRT_Star']

easy_configs = [
        [0.01, 0.33, 10],
        [-0.03, -0.30, 30],
        [-0.06, 0.23, -15],
        [0, 0.17, 83],
        [0.03, 0.38, -68],
        [0.04, 0.23, -18],
        [0.02, -0.29, -59],
        [0.02, 0.32, -47],
        [-0.01, 0.39, 78],
        [0.07, 0.38, 80],
        [0.01, 0.33, 15],
        [0.03, -0.20, 34],
        [-0.06, 0.31, -75],
        [0, -0.17, -53],
        [-0.03, 0.28, -78],
        [0.01, 0.33, 5],
        [-0.13, -0.30, 60],
        [-0.16, 0.33, -75],
        [0, 0.17, -63],
        [-0.03, 0.28, -28]
    ]

# undo_configs = [
#     [0.03, -0.20, 34],
#     [-0.06, 0.31, -75]
# ]

hard_configs = [
        [0, 0, -153],
        [0, 0, 117],
        [0, 0, -143],
        [0, 0, 155],
        [0, 0, -141],
        [0, 0, -120],
        [0, 0, 133],
        [0, 0, -156],
        [0, 0, 180],
        [0, 0, -131],
        [0.04, -0.23, -98],
        [-0.02, -0.21, -129],
        [0.02, -0.32, -144],
        [-0.01, 0.39, 168],
        [0.07, 0.28, 180],
        [0.04, -0.30, -92],
        [-0.02, -0.21, -139],
        [0.02, -0.12, -148],
        [-0.01, 0.34, 167],
        [0.07, 0.22, 170]
    ]

easy_configs_ = [[config[1], -config[0], config[2]] for config in easy_configs]
hard_configs_ = [[config[1], -config[0], config[2]] for config in hard_configs]


# Summarize all results for easy_configs
summary_results_easy = summarize_all_results(base_path, method_folders, easy_configs_)

# Summarize all results for hard_configs
summary_results_hard = summarize_all_results(base_path, method_folders, hard_configs_)

# Print the summary results for easy_configs
print("Results for Easy Configurations:" + "({} configs)".format(len(easy_configs)))
for method_folder, values in summary_results_easy.items():
    print(f"Method: {method_folder}")
    print(f"Avg MPD: {values['avg_mpd'][0]} += {values['avg_mpd'][1]}")
    print(f"Avg Chamfer: {values['avg_chamfer'][0]} += {values['avg_chamfer'][1]}")
    print(f"Avg Actions: {values['avg_actions'][0]} += {values['avg_actions'][1]}")
    print(f"Success: {values['success']} / {len(easy_configs)}")
    print("=" * 20)

# Print the summary results for hard_configs
print("Results for Hard Configurations:")
for method_folder, values in summary_results_hard.items():
    print(f"Method: {method_folder}")
    print(f"Avg MPD: {values['avg_mpd'][0]} += {values['avg_mpd'][1]}")
    print(f"Avg Chamfer: {values['avg_chamfer'][0]} += {values['avg_chamfer'][1]}")
    print(f"Avg Actions: {values['avg_actions'][0]} += {values['avg_actions'][1]}")
    print(f"Success: {values['success']} / {len(easy_configs)}")
    print("=" * 20)
