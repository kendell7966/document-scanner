##############################################################################
# 1/22/2025
# Kendell Cottam (kendellwc@protonmail.com)
# Course: Fundamentals of CV and IP
# Final Projects
# Project 3 - Document Scanner
# Locates a document from an image or video feed and removes the perspective
# and displays it aligned in another window.
##############################################################################

import cv2
import numpy as np

# NOTE: If document_path is an empty string then program will use video camera instead of loading an image file.
document_path = "scanned-form.jpg"  # non-empty path to use an image from a file
#document_path = ""                 # empty path to use video camera

document_aspect = 11 / 8.5
document_width = 500
document_height = int(document_width * document_aspect)


def set_destination_points():
    points = np.zeros((4,2), np.float32)
    points[0,:] = (0, 0)                            # top left
    points[1,:] = (document_width, 0)               # top right
    points[2,:] = (document_width, document_height) # bottom right
    points[3,:] = (0, document_height)              # bottom left
    return points


def extract_points(approxed_contours):
    points = np.zeros((4,2), np.float32)

    # Flatten the array containing the coordinates
    n = approxed_contours.ravel()
    i = 0
    k = 0
    for j in n:
        if i % 2 == 0:
            x = int(n[i])
            y = int(n[i + 1])
            points[k, :] = (x,y)
            k = k + 1
        i = i + 1

    return points


# Good (fast) code to arrange the order of the corner points,
# found at: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left

    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # print(pts)
    # print(s)
    # print(np.argmin(s))
    # print(np.argmax(s))
    # print(diff)
    # print(np.argmin(diff))
    # print(np.argmax(diff))

    # return the ordered coordinates
    return rect


def write_message(image, x, y, message):
    cv2.putText(image, message,
                (x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA)


def write_detection_failed_message(image):
    write_message(image, 10, 32, "Document not detected.")
    write_message(image, 10, 72, "Please ensure that all four")
    write_message(image, 10, 112, "corners are within the image.")


def update_scanner(document_source):
    document_source_copy = document_source.copy()

    gray = cv2.cvtColor(document_source, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)      # lower kernel size worked better for corner accuracy

    # NOTE: works fine for the provided image, may need to adjust for camera (lower for me)
    lower_threshold = 130

    _, thresholded = cv2.threshold(blurred, lower_threshold, 255, 0)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        write_detection_failed_message(document_source_copy)
        cv2.imshow(window_name_source, document_source)
        cv2.imshow(window_name_output, document_source_copy)
        return

    # Sort by area size, largest area first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # this contour is most likely the document, further checks below
    c = contours[0]
    perimeter = cv2.arcLength(c, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(c, epsilon, True)
    area = cv2.contourArea(c)
    #print(f"length: {len(approx)}, area: {area}")

    # Ensure there are 4 points and a large area
    min_area = 50000
    if len(approx) != 4 or area < min_area:
        write_detection_failed_message(document_source_copy)
        cv2.imshow(window_name_source, document_source)
        cv2.imshow(window_name_output, document_source_copy)
        return

    cv2.drawContours(document_source, [approx], -1, color=(0, 0, 255), thickness=2)

    points = extract_points(approx)
    points = order_points(points)

    for p in points:
        cv2.circle(document_source, (int(p[0]), int(p[1])), 5, (0,255,0), 2, cv2.LINE_AA)

    h, mask = cv2.findHomography(points, destination_points, cv2.RANSAC)
    document_source_copy = cv2.warpPerspective(document_source_copy, h, (document_width, document_height))

    cv2.imshow(window_name_source, document_source)
    cv2.imshow(window_name_output, document_source_copy)




window_name_source = "Image Source"
cv2.namedWindow(window_name_source, cv2.WINDOW_NORMAL)

window_name_output = "Document Output"
cv2.namedWindow(window_name_output, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name_output, document_width, document_height)

destination_points = set_destination_points()

if len(document_path) > 0:
    document = cv2.imread(document_path)
    cv2.resizeWindow(window_name_source, document.shape[1] // 2, document.shape[0] // 2)
    update_scanner(document)
    key = cv2.waitKey(0)
else:
    video = cv2.VideoCapture(0)
    isVideoOpen = video.isOpened()
    if not isVideoOpen:
        print("Error opening video stream")
    else:
        ret, frame = video.read()
        cv2.resizeWindow(window_name_source, frame.shape[1], frame.shape[0])

    while isVideoOpen:
        ret, frame = video.read()
        if ret:
            update_scanner(frame)

        key = cv2.waitKey(1)
        if key == 27:
            # Cleanup resources
            video.release()
            break


# Cleanup resources
cv2.destroyAllWindows()
