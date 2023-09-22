import argparse
import csv
import numpy as np
import ast
import pandas as pd

init_corners = np.array([[-0.294345, -0.18128946],
                            [ 0.29434484, -0.18128933],
                            [ 0.29434484, 0.18128933],
                            [-0.29434484, 0.18128951]])

def get_current_ori_change(rect1, rect2):
    # Calculate centroids of rectangles
    centroid1 = np.mean(rect1, axis=0)
    centroid2 = np.mean(rect2, axis=0)

    # Center points by subtracting centroids
    centered1 = rect1 - centroid1
    centered2 = rect2 - centroid2

    # Calculate covariance matrix
    covariance_matrix = np.dot(centered1.T, centered2)

    # Perform singular value decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Calculate rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)

    # Calculate angle of rotation in radians
    angle_radians = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)

    # print("Rotation angle (degrees):", angle_degrees)
    return angle_degrees

# Define a function to calculate the center and orientation values
# def calculate_center_orientation(row):
#     # Split the result_points string into separate values
#     curr_corners = ast.literal_eval(row['result_points'])
#     center = np.mean(curr_corners, axis=0)

#     ori_change = get_current_ori_change(init_corners, curr_corners)

# Define a function to calculate the center and orientation values
def calculate_center_orientation(row):
    # Split the result_points string into separate values
    curr_corners = ast.literal_eval(row['result_points'])
    center = np.mean(curr_corners, axis=0)

    init_corners = np.array([[-0.294345, -0.18128946],
                            [ 0.29434484, -0.18128933],
                            [ 0.29434484, 0.18128933],
                            [-0.29434484, 0.18128951]])
    ori_change = get_current_ori_change(init_corners, curr_corners)

    # Calculate the orientation value as the angle in degrees between the x-axis and the orientation vector

    return pd.Series({"delta_x": center[0], "delta_y": center[1], "delta_r": ori_change})

def process_dataframe(csv_file, output_file):

    samples_db = pd.read_csv(csv_file)
    
    # samples_db = process_dataframe(samples_db)
    # print("Num of rows: ", len(df))
    
    # Apply the function to each row of the DataFrame and add the resulting columns to the DataFrame
    samples_db[["delta_x", "delta_y", "delta_r"]] = samples_db.apply(calculate_center_orientation, axis=1)

    samples_db = samples_db.drop(columns=["result_points"])
    samples_db = samples_db.drop(columns=["deformation"])

    samples_db.to_csv(output_file, index=False)

def process_csv_file(csv_file, output_file):
    # Open the CSV file for reading
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Define the output field order
        fieldnames = ['delta_x', 'delta_y', 'delta_r', 'push_x', 'push_y', 'push_ori', 'trans_x', 'trans_y', 'rot']

        # Open the output file for writing
        with open(output_file, 'w') as outfile:
            writer = csv.writer(outfile, delimiter=' ')

            # Write the header row
            # writer.writerow(fieldnames)

            # Iterate over the input rows and write the transformed rows to the output file
            for row in reader:
                # Extract the relevant columns and apply the conversion function
                delta_x, delta_y, delta_r = row['delta_x'], row['delta_y'], row['delta_r']
                push_x, push_y, push_ori = row['push_x'], row['push_y'], row['push_ori']
                trans_x, trans_y, rot = row['trans_x'], row['trans_y'], row['rot']

                # Write the transformed row to the output file
                writer.writerow([delta_x, delta_y, delta_r, push_x, push_y, push_ori, trans_x, trans_y, rot])
                # outfile.write('\n')

def process_to_data(csv_file, output_file):
    # Open the CSV file for reading
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Define the output field order
        # push_x,push_y,push_ori,rot,trans_x,trans_y,deformation,result_points
        origin_names = ["push_x","push_y","push_ori","rot","trans_x","trans_y","deformation","result_points"]

        # Open the output file for writing
        with open(output_file, 'w') as outfile:
            writer = csv.writer(outfile, delimiter=' ')

            fieldnames = ['delta_x', 'delta_y', 'delta_r', 'push_x', 'push_y', 'push_ori', 'trans_x', 'trans_y', 'rot']

            # Write the header row
            # writer.writerow(fieldnames)

            # Iterate over the input rows and write the transformed rows to the output file
            for row in reader:
                # Extract the relevant columns and apply the conversion function
                # delta_x, delta_y, delta_r = row['delta_x'], row['delta_y'], row['delta_r']

                # Split the result_points string into separate values
                curr_corners = ast.literal_eval(row['result_points'])
                center = np.mean(curr_corners, axis=0)

                # get change in orientation and translation
                delta_r = get_current_ori_change(init_corners, curr_corners)
                delta_x = center[0]
                delta_y = center[1]

                # parse result_points

                push_x, push_y, push_ori = row['push_x'], row['push_y'], row['push_ori']
                trans_x, trans_y, rot = row['trans_x'], row['trans_y'], row['rot']

                # Write the transformed row to the output file
                writer.writerow([delta_x, delta_y, delta_r, push_x, push_y, push_ori, trans_x, trans_y, rot])
                # outfile.write('\n')

if __name__ == '__main__':
    # # Parse command-line arguments
    # parser = argparse.ArgumentParser(description='Convert CSV file to text file')
    # parser.add_argument('input_file', help='input CSV file')
    # parser.add_argument('output_file', help='output text file')
    # args = parser.parse_args()

    # Process the CSV file and write the output to the text file
    process_dataframe("deform_data_small_step_2cm.csv", "deform_data_small_step_2cm_ctl.csv")