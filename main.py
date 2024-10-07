import cv2
import numpy as np
import matplotlib.image as mplimg
import glob
import re
import os


# to get the coordinates
def calculate_coordinates(image, line_params):
    try :
        slope, intercept = line_params
    except:
        slope, intercept = (1,1)
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


# calculate the average slope intercepts
def average_slope_intercept(image, lines):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    left_avg = np.average(left_lines, axis=0)
    right_avg = np.average(right_lines, axis=0)
    left_line = calculate_coordinates(image, left_avg)
    right_line = calculate_coordinates(image, right_avg)
    return np.array([left_line, right_line])


# apply canny algorithm
def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_result = cv2.Canny(blur, 50, 150)
    return canny_result


# get the interception points
def get_intersection_point(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def determinant(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = determinant(xdiff, ydiff)
    if div == 0:
        print('Lines do not intersect.')

    d = (determinant(*line1), determinant(*line2))
    x = determinant(d, xdiff) / div
    y = determinant(d, ydiff) / div
    return x, y


# draw the lines and circle
def draw_lines_and_circle(image, lines):
    line_image = np.zeros_like(image)
    A = [lines[0][0], lines[0][1]]
    B = [lines[0][2], lines[0][3]]
    C = [lines[1][0], lines[1][1]]
    D = [lines[1][2], lines[1][3]]

    coordinates = get_intersection_point((A, B), (C, D))

    if lines is not None:
        cv2.line(line_image, A, (int(coordinates[0]), int(coordinates[1])), (255, 0, 0), 3)
        cv2.line(line_image, C, (int(coordinates[0]), int(coordinates[1])), (255, 0, 0), 3)

    cv2.circle(line_image, (int(coordinates[0]), int(coordinates[1])), 8, (0, 255, 0), 4)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([[(0, height), (200, 290), (200, 320), (width, height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# mark the lane lines
def mark_lines(image):
    canny_result = apply_canny(image)
    cropped_image = region_of_interest(canny_result)

    # Define the Hough transform parameters
    rho = 2
    theta = np.pi / 180
    threshold = 70
    min_line_length = 30
    max_line_gap = 20

    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, min_line_length, max_line_gap)
    averaged_lines = average_slope_intercept(image, lines)
    line_image = draw_lines_and_circle(image, averaged_lines)
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combo_image


# create the video
def create_video(output_folder):

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    img_array = []
    filename_array = []
    for filename in glob.glob(output_folder + '/*.jpg'):
        filename_array.append(filename)

    filename_array.sort(key=natural_keys)

    for image in filename_array:
        img = cv2.imread(image)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    is_exist = os.path.exists(output_folder+'_Video')
    if not is_exist:
        os.mkdir(output_folder+'_Video')

    out = cv2.VideoWriter(output_folder+'_Video'+'/OutputVideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def main(input_folder, output_folder):
    paths = []
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    for file in files:
        path = input_folder + '/' + file
        paths.append(path)

    is_exist = os.path.exists(output_folder)
    if not is_exist:
        os.mkdir(output_folder)

    out_img_paths = glob.glob(output_folder + '/*.jpg')
    for i, del_image in enumerate(out_img_paths):
        os.remove(del_image)

    for img_pth in enumerate(paths):
        image_path = img_pth[1]
        print(image_path)
        image = mplimg.imread(image_path)
        lane_image = np.copy(image)
        result = mark_lines(lane_image)
        mplimg.imsave(output_folder + '/' + image_path[12:-4] + '_processed.jpg', result)

    create_video(output_folder)


input_folders = ['TestVideo_1', 'TestVideo_2']
output_folders = ['processed_outputs_1', 'processed_outputs_2']

for i in [0, 1]:
    main(input_folders[i], output_folders[i])
