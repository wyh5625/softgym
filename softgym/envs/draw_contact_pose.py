import matplotlib.pyplot as plt
import numpy as np
import math
import re

model_folder = "./data/model/"


def read_corner_idx(model_name):
    shape_path = model_folder + model_name + ".txt"

    # Open the text file for reading
    with open(shape_path, 'r') as file:
        # Initialize an empty list to store the first numbers
        particle_idx = []
        
        # Read the file line by line
        for line in file:
            # Split the line by whitespace to get individual values
            values = line.split()
            if values:
                # Extract the first number from the values
                first_number = int(values[0])
                # Append the first number to the list
                particle_idx.append(first_number)

    # Print the list of first numbers
    return particle_idx


# Calculate the contact poses for each side of the polygon
def calculate_contact_poses(vertices, pl, concave_corners):
    contact_poses = []
    concave_contact_poses = []  # List to store concave contact poses
    num_vertices = len(vertices)

    for i in range(num_vertices):
        current_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % num_vertices]

        # Check if current_vertex or next_vertex is in concave_corners
        is_current_concave = i in concave_corners
        is_next_concave = (i + 1) % num_vertices in concave_corners

        # Calculate the direction vector for the current side
        direction = (next_vertex[0] - current_vertex[0], next_vertex[1] - current_vertex[1])
        magnitude = np.linalg.norm(direction)
        normalized_direction = (direction[0] / magnitude, direction[1] / magnitude)  # Calculate normalized direction

        # Calculate contact poses along the side
        for j in range(3):
            if j == 0:
                x = current_vertex[0]
                y = current_vertex[1]
            elif j == 1:
                center = ((current_vertex[0] + next_vertex[0]) / 2, (current_vertex[1] + next_vertex[1]) / 2)
                x = center[0] - 0.5 * pl * normalized_direction[0]
                y = center[1] - 0.5 * pl * normalized_direction[1]
            else:
                x = next_vertex[0] - pl * normalized_direction[0]
                y = next_vertex[1] - pl * normalized_direction[1]

            contact_pose = (x, y, normalized_direction)  # Include normalized direction
            contact_poses.append(contact_pose)

            # Check if the contact pose is concave and add it to concave_contact_poses
            if is_current_concave or is_next_concave:
                concave_contact_poses.append(contact_pose)

    return contact_poses, concave_contact_poses


# Function to find concave corners and return their indices
def find_concave_corners(vertices):
    concave_corners = []
    num_vertices = len(vertices)

    for current in range(num_vertices):
        prev = current - 1 if current > 0 else num_vertices - 1
        next = (current + 1) % num_vertices

        # Convert the current vertex to a list, append 0, and then convert back to a tuple
        point0 = tuple(list(vertices[current]) + [0])
        point1 = tuple(list(vertices[prev]) + [0])
        point2 = tuple(list(vertices[next]) + [0])

        dir0 = np.array(point0) - np.array(point1)
        dir0 = dir0 / np.linalg.norm(dir0)

        dir1 = np.array(point2) - np.array(point0)
        dir1 = dir1 / np.linalg.norm(dir1)

        dir2 = np.cross(dir0, dir1)
        dir2 = dir2 / np.linalg.norm(dir2)

        dot = np.dot(dir2, np.array([0, 0, 1]))  # Assuming a 2D plane with normal (0, 0, 1)
        if dot < 0.0:
            concave_corners.append(current)  # Append the index of the concave corner

    return concave_corners

def show(model_name):
    shape_path = model_folder + model_name + ".txt"

    # Read vertices from 'shape.txt' and store them in a list
    vertices = []
    with open(shape_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                x, y = float(parts[1]), float(parts[2])
                vertices.append((x, y))

    # Set the contact pose length (pl)
    pl = 0.02

    # Find concave corners
    concave_corners = find_concave_corners(vertices)

    # Calculate contact poses and concave contact poses
    contact_poses, concave_contact_poses = calculate_contact_poses(vertices, pl, concave_corners)

    # Plot the shape contour
    x, y = zip(*vertices)
    plt.plot(x + (x[0],), y + (y[0],), marker='o', label='Shape Contour', markersize=10)


    # Plot contact poses
    for i, pose in enumerate(contact_poses):
        label = f'Contact Pose {i + 1}'
        normalized_direction = pose[2]  # Extract normalized direction

        # Use red color for concave contact poses and blue for convex contact poses
        color = 'red' if pose in concave_contact_poses else 'blue'

        plt.plot([pose[0], pose[0] + pl * normalized_direction[0]], [pose[1], pose[1] + pl * normalized_direction[1]],
                label=label, color=color, linewidth=4.0)
        plt.text(pose[0] + pl / 2, pose[1] + pl / 2, f'{i + 1}', horizontalalignment='center', verticalalignment='center')


    # Invert the y-axis to show it upside down
    plt.gca().invert_yaxis()

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()


    # Show the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()



# Function to calculate the angle between two vectors
def calculate_angle(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))
    cosine_theta = dot_product / (magnitude1 * magnitude2)
    cosine_theta = max(min(cosine_theta, 1), -1)
    # print("cosine_theta:", cosine_theta)
    return math.degrees(math.acos(cosine_theta))


def sort_boundary(corner_particles, boundary_edges):
    sorted_boundary = {}
    sorted_corners = []
    visited_vertices = set()

    # Find a vertex that has two boundary edges
    start_vertex = None
    for vertex in corner_particles:
        connected_edges = [edge for edge in boundary_edges if vertex in edge]
        if len(connected_edges) == 2:
            start_vertex = vertex
            break

    if start_vertex is None:
        # If no vertex with two boundary edges is found, return an empty list
        return []

    # Create a dictionary to store boundary segments for each corner
    current_segment = []

    # Iterate through the boundary edges in order
    current_vertex = start_vertex
    current_corner = start_vertex
    while True:
        # sorted_boundary.append(current_vertex)
        # current_segment.append(current_vertex)
        visited_vertices.add(current_vertex)

        # Check if the current vertex is a corner
        if current_vertex in corner_particles:
            sorted_corners.append(current_vertex)

        # Find the next vertex connected to the current vertex
        next_vertex = None
        for edge in boundary_edges:
            if current_vertex in edge:
                next_vertex = edge[1] if edge[0] == current_vertex else edge[0]
                if next_vertex not in visited_vertices:
                    break
                else:
                    next_vertex = None

        if next_vertex is not None:
            current_vertex = next_vertex
        else:
            # If no unvisited neighbors are left, break the loop
            sorted_boundary[current_corner] = current_segment
            break

        if current_vertex not in corner_particles:
            current_segment.append(current_vertex)
        elif current_vertex is not start_vertex:
            sorted_boundary[current_corner] = current_segment
            current_corner = current_vertex
            current_segment = []

    return sorted_corners, sorted_boundary

def find_corner_and_segments(model_path):
    # Read the OBJ file and extract vertices and faces
    vertices = []  # List to store vertices
    faces = []  # List to store faces (vertex indices)

    with open(model_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertex = [float(coord) for coord in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(index) for index in re.findall(r"\d+", line)]
                faces.append(face)

    # Create a set to store boundary vertices
    boundary_vertices = set()

    # Create a list to store boundary edges
    boundary_edges = []

    # Iterate through faces to find boundary edges
    for face in faces:
        face = [face[0], face[3], face[6]]
        for i in range(3):
            edge1, edge2 = (face[i], face[(i + 1) % 3]), (face[(i + 1) % 3], face[i])
            if edge1 in boundary_edges:
                # This edge is no longer a boundary edge
                boundary_edges.remove(edge1)
            elif edge2 in boundary_edges:
                # This edge is no longer a boundary edge
                boundary_edges.remove(edge2)
            else:
                # This edge is a boundary edge
                boundary_edges.append(edge1)

    # boundary_edges: [(191, 2160), ...]

    # Extract boundary vertices from boundary edges
    for edge in boundary_edges:
        boundary_vertices.add(edge[0])
        boundary_vertices.add(edge[1])

    # boundary_vertices:  {1, 2, 3, ... }

    # Extract the coordinates of boundary vertices
    boundary_coordinates = [vertices[i - 1] for i in boundary_vertices]

    threshold_angle = 15
    corner_particles = set()

    angles = []
    # Iterate through each boundary particle
    for particle in boundary_vertices:
        connected_edges = [edge for edge in boundary_edges if particle in edge]
        if len(connected_edges) == 2:
            edge1, edge2 = connected_edges
            # Find the vectors representing the two edges
            vector1 = (vertices[edge1[1] - 1][0] - vertices[edge1[0] - 1][0],
                       vertices[edge1[1] - 1][2] - vertices[edge1[0] - 1][2])
            vector2 = (vertices[edge2[1] - 1][0] - vertices[edge2[0] - 1][0],
                       vertices[edge2[1] - 1][2] - vertices[edge2[0] - 1][2])

            # Calculate the angle between the vectors
            angle = calculate_angle(vector1, vector2)

            if angle > threshold_angle:
                angles.append(angle)
                corner_particles.add(particle)

    # Sort the corner particles
    sorted_corners, sorted_boundary = sort_boundary(corner_particles, boundary_edges)

    # ### 1. Plot corners ###
    #
    # # Convert corner particles to x and y coordinates
    # corner_x = [vertices[i - 1][0] for i in sorted_corners]
    # corner_y = [vertices[i - 1][2] for i in sorted_corners]
    # plt.scatter(corner_x, corner_y, label="Corner Particles", color="red")
    #
    # # Add labels for corner particles
    # for i, (x, y) in enumerate(zip(corner_x, corner_y)):
    #     plt.text(x, y, str(i), fontsize=12, ha='center', va='bottom', color='red')
    #
    # ### 2. Plot boundary segments ###
    #
    # # Plot the sorted boundary segments
    # print(sorted_boundary)
    # for corner_particle in sorted_boundary.keys():
    #     segment = sorted_boundary[corner_particle]
    #     # print(segment)
    #
    #     x_values = [vertices[i - 1][0] for i in segment]
    #     y_values = [vertices[i - 1][2] for i in segment]
    #     plt.scatter(x_values, y_values, label="Segment Particles", color="blue")
    #
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # # plt.legend()
    # plt.title("Boundary and Corner Particles")
    # plt.grid(True)
    # plt.show()

    return sorted_corners, sorted_boundary

# save corner, segment particles
def save_sorted_boundary_segments_to_file(sorted_boundary_segments, output_file):
    with open(output_file, 'w') as file:
        for corner, segments in sorted_boundary_segments.items():
            line = ' '.join(map(str, [corner] + [x for x in segments]))
            file.write(line + '\n')

def get_corner_and_segment(model_name):
    model_path = model_folder + model_name + ".obj"
    sorted_corners, sorted_boundary = find_corner_and_segments(model_path)
    output_path = model_folder + model_name + ".txt"
    save_sorted_boundary_segments_to_file(sorted_boundary, output_path)

def get_vertice_particle_mapping(model_name):
    model_path = model_folder + model_name + ".obj"

    # Define a dictionary to store unique vertex IDs
    unique_vertex_ids = {}
    unique_id = 0

    found_UV_id = []

    # Open and read the OBJ file
    with open(model_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('f '):  # Parse face definitions (lines starting with 'f')
                parts = line.split()
                if len(parts) >= 4:
                    # Extract vertex indices from face definition
                    indices = [(int(part.split('/')[0]), int(part.split('/')[1])) for part in parts[1:]]
                    
                    # Check and update unique vertex IDs
                    for v_id, uv_id in indices:
                        if uv_id not in found_UV_id:
                            found_UV_id.append(uv_id)
                            unique_vertex_ids[v_id] = unique_id
                            unique_id += 1

    # unique_vertex_ids now contains unique vertex IDs for face definitions
    return unique_vertex_ids

def read_sorted_boundary_segments_from_file(model_name):
    # the actual indices of particles are from 0 to n-1
    input_file = model_folder + model_name + ".txt"
    sorted_boundary_segments = {}
    with open(input_file, 'r') as file:
        for line in file:
            data = line.strip().split()
            corner = int(data[0])
            segments = [int(x) for x in data[1:]]
            sorted_boundary_segments[corner] = segments
    return sorted_boundary_segments

def get_vertice_particle_mapping(model_name):
    file_path = model_folder + model_name + ".obj"

    # Define a dictionary to store unique vertex IDs
    unique_vertex_ids = {}
    unique_id = 0

    found_UV_id = []

    # Open and read the OBJ file
    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('f '):  # Parse face definitions (lines starting with 'f')
                parts = line.split()
                if len(parts) >= 4:
                    # Extract vertex indices from face definition
                    indices = [(int(part.split('/')[0]), int(part.split('/')[1])) for part in parts[1:]]
                    
                    # Check and update unique vertex IDs
                    for v_id, uv_id in indices:
                        if uv_id not in found_UV_id:
                            found_UV_id.append(uv_id)
                            unique_vertex_ids[v_id] = unique_id
                            unique_id += 1

    # unique_vertex_ids now contains unique vertex IDs for face definitions
    return unique_vertex_ids

def get_contact_poses(particle_pos, sorted_boundary_segments, pusher_length):
    # there are three contact pose along each segment: 1. left-end touch start point, right-end touch the farest point along the segment
    # 2. center touch middle point of segment, parallel to connection between ends of segment  
    # 3. right-end touch end point, left-end touch the farest point along the segment
    contact_poses = []
    for corner, segments in sorted_boundary_segments.items():

        ### 1. find the 1st contact pose ###

        start_point = particle_pos[corner]
        # end point is the farest point along the segment
        print("corner: ", start_point)
        end_point = particle_pos[segments[-1]]
        
        # loop through the segment to find the farest point
        for segment in segments[::-1]:
            if np.linalg.norm(particle_pos[segment] - start_point) <= pusher_length:
                end_point = particle_pos[segment]
                break
        
        # calculate the direction vector for the current side
        direction = end_point - start_point
        magnitude = np.linalg.norm(direction)
        normalized_direction = direction / magnitude
        p_o = start_point + 0.5 * pusher_length * normalized_direction

        # get orientation of pusher
        vec = end_point - start_point
        # find artan of vecA
        p_ori = math.atan2(vec[1], vec[0])
        contact_poses.append([p_o[0], p_o[1], p_ori])

        ### 2. find the 2nd contact pose ###

        # find the middle point of the segment
        p_o = particle_pos[segments[len(segments)//2]]

        # find the orientation of the pusher
        vec = particle_pos[segments[-1]] - particle_pos[segments[0]]

        # find artan of vecA
        p_ori = math.atan2(vec[1], vec[0])
        contact_poses.append([p_o[0], p_o[1], p_ori])


        ### 3. find the 3rd contact pose ###

        end_point = particle_pos[segments[-1]]
        # loop through the segment to find the farest start point
        for segment in segments:
            if np.linalg.norm(particle_pos[segment] - end_point) <= pusher_length:
                start_point = particle_pos[segment]
                break
        
        direction = start_point - end_point
        magnitude = np.linalg.norm(direction)
        normalized_direction = direction / magnitude
        p_o = end_point + 0.5 * pusher_length * normalized_direction


        # get orientation of pusher
        vec = end_point - start_point
        # find artan of vecA
        p_ori = math.atan2(vec[1], vec[0])
        contact_poses.append([p_o[0], p_o[1], p_ori])
        
    return contact_poses
        
