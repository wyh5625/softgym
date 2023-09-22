import numpy as np

def get_current_ori_change2(rect1, rect2):

        # Sample rectangle points (4 points per rectangle)
        rectangle1 = np.array(rect1)[:, [0,2]]
        
        rectangle2 = np.array(rect2)

        print(rectangle1)
        print(rectangle2)

        # Calculate centroids of rectangles
        centroid1 = np.mean(rectangle1, axis=0)
        centroid2 = np.mean(rectangle2, axis=0)         

        # Center points by subtracting centroids
        centered1 = rectangle1 - centroid1
        centered2 = rectangle2 - centroid2

        print("Centered:")      
        print(centered1)
        print(centered2)

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

def get_current_ori_change(rect1, rect2):
    # Sample rectangle points (4 points per rectangle)
    # rectangle1 = np.array(rect1)[:, [0, 2]]
    #
    # rectangle2 = np.array(rect2)[:, [0, 2]]

    # Calculate centroids of rectangles
    centroid1 = np.mean(rect1, axis=0)
    centroid2 = np.mean(rect2, axis=0)

    # Center points by subtracting centroids
    centered1 = rect1 - centroid1
    centered2 = rect2 - centroid2

    print("Centered:")      
    print(centered1)
    print(centered2)

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

init_corners = np.array([[-0.294345, -0.18128946],
 [ 0.29434484, -0.18128933],
 [ 0.29434484, 0.18128933],
 [-0.29434484, 0.18128951]])

init_corners = np.array([[-0.294, -0.181],
 [ 0.294,  -0.181],
 [ 0.294,  0.181],
 [-0.294,  0.181]]
)


curr_corners = np.array([[-1.0586987733840942, -1.001754879951477], [-0.5069950222969055, -0.9002631902694702], [-0.5731560587882996, -0.5462145209312439], [-1.118407130241394, -0.6556845903396606]])
curr_corners = np.array([[-1.059, -1.002],
 [-0.507, -0.9],
 [-0.573, -0.546],
 [-1.118, -0.656]])

ori_change = get_current_ori_change(init_corners, curr_corners)
print("Cloth rotated: ", ori_change)