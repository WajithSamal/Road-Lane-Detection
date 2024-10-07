import cv2
import numpy as np

image = cv2.imread('TestVideo_1/Right_0.bmp', cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread('TestVideo_1/Right_0.bmp')

# Define the region of interest (ROI) vertices
height, width = image.shape
roi_bottom_left = (0, height)
roi_bottom_right = (width, height)
roi_top_left = (0, height // 2 + 50)
roi_top_right = (width, height // 2 + 50)
roi_vertices = np.array([[roi_bottom_left, roi_top_left, roi_top_right, roi_bottom_right]], dtype=np.int32)

# Create a mask with zeros and fill the ROI with ones
mask = np.zeros_like(image)
cv2.fillPoly(mask, roi_vertices, 255)

# Apply the mask to the image
image = cv2.bitwise_and(image, mask)


def display_image(img, name='image', wait=1):
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def apply_sobel(img, blur=1):
    if blur:
        new_img = cv2.GaussianBlur(img, (5, 5), 0)
    else:
        new_img = img
    sobel_x = cv2.Sobel(new_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(new_img, cv2.CV_64F, 0, 1, ksize=3)

    # Combine the results from Sobel x and Sobel y
    sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Convert the result to uint8
    sobel_combined = np.uint8(sobel_combined)
    return sobel_combined


edges = apply_sobel(image)
#edges= cv2.Canny(image, 50, 150, apertureSize=3)

threshold_value = 200
_, thresholded = cv2.threshold(edges, threshold_value, 255, cv2.THRESH_BINARY)

lines = cv2.HoughLinesP(thresholded, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=40,
                             lines=np.array([]))
filtered_lines=[]
for line in lines:
    x1, y1, x2, y2 = line[0]
    slope = (y2-y1)/x2-x1
    if abs(slope)>0.5:
        filtered_lines.append(line)

for line in filtered_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the result
cv2.imshow('Lane Detection Result', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
