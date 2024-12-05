import pygame
from pygame.locals import *
import sys
import numpy as np


model_folder = "./data/model/"

# 窗口尺寸
window_size = (600, 600)

# 读取OBJ文件并提取顶点的x和y坐标
def read_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                # 归一化x和y坐标到[0, 1]范围
                x = (x + 1) / 2
                z = (z + 1) / 2
                vertices.append((x, z))
    return vertices

# 找到最近的点
def find_nearest_point(mouse_pos, vertices):
    mouse_x, mouse_y = mouse_pos
    nearest_point = None
    min_distance = float('inf')
    for i, vertex in enumerate(vertices):
        x, y = vertex
        x_pixel = x*window_size[0]
        y_pixel = y*window_size[1]
        distance = np.sqrt((x_pixel - mouse_x) ** 2 + (y_pixel - mouse_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = (i+1, vertex)
    return nearest_point

# 保存点击的点到文件
def save_points_to_file(points, file_path):
    with open(file_path, 'w') as file:
        for idx, point in points:
            x, y = point
            file.write(f"{idx} {x} {y}\n")


def main(model_name="pants"):
    # 初始化Pygame
    pygame.init()



    # 创建窗口
    screen = pygame.display.set_mode(window_size)

    # 读取OBJ文件并提取顶点的x和y坐标
    file_name = model_name + ".obj"
    obj_file_path = model_folder + file_name
    vertices = read_obj(obj_file_path)

    inMap = get_vertice_particle_mapping(obj_file_path)

    # Corners txt
    corners_data_file = model_folder + model_name + ".txt"

    # 存储点击的点
    clicked_points = []

    print(inMap)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    mouse_pos = pygame.mouse.get_pos()
                    idx, nearest_point = find_nearest_point(mouse_pos, vertices)
                    print("the idx: ", idx)
                    idx = inMap[idx]    # Convert to particle index
                    if nearest_point:
                        # convert to original vertex
                        origin_point = (nearest_point[0]*2 - 1, nearest_point[1]*2 - 1)
                        clicked_points.append((idx, origin_point))
                        print(f"Clicked point {idx}: {origin_point}")
                elif event.button == 3:  # 右键点击
                    running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:  # 按下空格键保存点并退出
                    save_points_to_file(clicked_points, corners_data_file)
                    running = False

        # 清空屏幕
        screen.fill((255, 255, 255))

        # 在窗口上绘制点
        for vertex in vertices:
            x, y = vertex
            # 将点坐标映射到窗口大小
            x *= window_size[0]
            y *= window_size[1]
            pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 2)

        # 刷新窗口
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
