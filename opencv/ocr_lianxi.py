import sys
import time
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt, gridspec
from paddleocr import PaddleOCR
from difflib import SequenceMatcher

# Define a function to extract rectangular images of text blocks (note that contours need to be closed, otherwise blank images will be extracted)
def extract_contour_as_rectangular_image(contour, image):
    # Get the minimum enclosing rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Extract the rectangular region (ROI) from the original image
    roi = image[y:y + h, x:x + w]

    # Create a mask of the same size as the ROI, initially all black (0 indicates fully transparent area)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Move the contour to the coordinate system relative to the ROI
    contour_shifted = contour - [x, y]

    # Draw the contour on the mask image, filling the inside with white to indicate the region of interest
    cv2.drawContours(mask, [contour_shifted], -1, 255, thickness=cv2.FILLED)

    # Create a white background image of the same size as the ROI
    result = np.ones_like(roi, dtype=np.uint8) * 255

    # Use masks to extract pixels inside contours from ROIs
    roi_with_mask = cv2.bitwise_and(roi, roi, mask=mask)

    # Paste the extracted outline area onto a white background and put the non-white part of the outline into the result
    np.copyto(result, roi_with_mask, where=(mask == 255))
    return result

# Define a function to sort the contours
def sort_contours(contours, delta=30):
    # Sort contours by y-coordinate first, then by x-coordinate
    contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))

    avg_y_coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        avg_y = y + h / 2  # 计算 y 轴的中心点
        avg_y_coordinates.append(avg_y)

    sorted_contours = []
    need_newline = []
    current_row = []

    # Group and sort by average y-coordinate
    for i, contour in enumerate(contours):
        avg_y = avg_y_coordinates[i]
        if len(current_row) == 0:
            # The current line is empty, add the first element directly
            current_row.append((contour, avg_y))  # 将轮廓和 avg_y 一起存储
        else:
            # Determine if they are on the same line
            if abs(avg_y - current_row[0][1]) <= delta:
                # Add the current line if it is on the same line
                current_row.append((contour, avg_y))
            else:
                # If not on the same line then sort current line by x-coordinate
                current_row = sorted(current_row, key=lambda item: cv2.boundingRect(item[0])[0])

                # Add to the results array
                sorted_contours.extend([item[0] for item in current_row])

                # Line breaks are required for the first block of text, not the rest.
                need_newline.append(True)
                need_newline.extend([False] * (len(current_row) - 1))

                # Clear the current line and start processing a new one
                current_row = [(contour, avg_y)]

    # Process the last line
    if len(current_row) > 0:
        # Sort current row by x-coordinate
        current_row = sorted(current_row, key=lambda item: cv2.boundingRect(item[0])[0])

        # Add to the results array
        sorted_contours.extend([item[0] for item in current_row])

        # Line breaks are required for the first block of text, not the rest.
        need_newline.append(True)
        need_newline.extend([False] * (len(current_row) - 1))
    return need_newline,sorted_contours

# Define a function to detect the correction angle and rotate the text block
def detect_and_rotate_image(image):

    # Use Canny edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Detecting straight lines using the Hough transform
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=100,
                            minLineLength=50,
                            maxLineGap=10)

    angles = []  # 存储直线的角度

    # Make sure a straight line is detected
    if lines is not None:
        # Iterate over all lines detected
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate the angle of each line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

    # Calculate the rotation angle and take the average as the rotation angle
    if len(angles) > 0:
        median_angle = np.median(angles)
    else:
        # No rotation if no straight line detected
        median_angle = 0

    # Get the height and width of the image
    (h, w) = image.shape[:2]
    # Calculate the center of rotation
    center = (w // 2, h // 2)

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Calculate the boundaries of the rotated image to fit the whole image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Boundary size of the new image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to fit the new boundary
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform affine rotation and fill the boundary with white color
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

    # Return the rotated image
    return rotated_image

# Define function to compute the intersection-conjugation ratio (IoU) of two bounding boxes
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Calculate the concatenation plane
    union = w1 * h1 + w2 * h2 - intersection

    if union == 0:
        # Avoid dividing by zero
        return 0
    iou = intersection / union
    return iou

# Define a function that calculates the percentage of black pixels in the outline area to filter the text block
def compute_black_pixel_ratio(image, contour):

    # Get the outer rectangular box of the outline
    x, y, w, h = cv2.boundingRect(contour)

    # Extract the contour area
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # Calculate the total number of pixels and the number of black pixels in the area
    total_pixels = w * h
    if total_pixels == 0:
        return 0
    # Calculate the number of black pixels
    black_pixels = np.sum(roi == 0)
    if black_pixels > total_pixels:
        # If the number of black pixels is greater than the total number of pixels, the font is white
        black_pixels = np.sum(roi == 255)

    # Calculate the proportions of the font
    black_pixel_ratio = black_pixels / total_pixels
    return black_pixel_ratio

# Define functions, image contrast and brightness enhancement
def image_enhancement(image):

    # Convert images to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate image contrast using the standard deviation of a grayscale image (the higher the standard deviation, the higher the contrast)
    contrast_std = np.std(gray)

    # Adaptive adjustment of contrast enhancement multiplier, no contrast increase when standard deviation is greater than 30
    if contrast_std > 30:
        # Contrast is already high enough without adding contrast
        contrast_factor = 1.0
    else:
        contrast_factor = 1.2
    image_contrasted = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)  # alpha是对比度系数

    # Calculate the variance of the Laplace transform to get the image sharpness
    clear = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Calculate the average brightness value of the image
    mean_brightness = np.mean(gray)

    # Brightness Enhancement, dynamically adjusts the Brightness Enhancement Multiplier using Sharpness and Average Brightness, and limits the Brightness Enhancement Multiplier to [1, 2]
    brightness_factor = max(min(round(clear / 2000, 1), 2), 1)

    # If the average luminance is already high (e.g. > 180), limit the luminance enhancement multiplier to avoid over-enhancement
    if mean_brightness > 180:
        brightness_factor = 0

    # Brightness enhancement only for eligible areas
    image_brightened = cv2.convertScaleAbs(image_contrasted, alpha=1, beta=brightness_factor * 50)
    return image_brightened

# Defining mouse click events
def mouse_click(event):
    global points
    global warped
    if event.inaxes:

        # Record the location of the click
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))

        # Duplicate the image and draw points
        temp_img = image.copy()
        for i, point in enumerate(points):
            cv2.circle(temp_img, point, 10, (0, 255, 0), -1)

        # Update the display in Matplotlib
        ax.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
        plt.draw()

        # When four points have been clicked, perform a perspective transformation
        if len(points) == 4:
            pts = np.array(points, dtype="float32")
            rect = order_points(pts)

            # Drawing a quadrilateral
            for i in range(4):
                pt1 = tuple(rect[i].astype(int))
                pt2 = tuple(rect[(i + 1) % 4].astype(int))
                cv2.line(temp_img, pt1, pt2, (0, 255, 0), 2)

            # Update to show images with rectangular boxes
            ax.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
            plt.draw()
            # Close the current window
            plt.close()
            # Perform a four-point perspective transformation
            warped, dst = four_point_transform(image, rect)

            # Create a new Matplotlib window to display the results
            plt.figure(figsize=(18, 9))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image with Points')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.title('Transformed Image')
            plt.axis('off')

            # Show results
            plt.show()

# Define functions to perform perspective transformations
def four_point_transform(image, rect):

    # The corner points of the target rectangle after transformation:
    (tl, tr, br, bl) = rect
    # Convert corner points to a form suitable for plotting (polygon point sets should be of integer type)
    pts = np.array([tl, tr, br, bl], dtype=np.int32)

    # Calculating width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = int(max(widthA, widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = int(max(heightA, heightB))

    # Get the position of the target after transformation
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Perform perspective transform
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped,dst

# Define function to automatically sort corner points in the order top left, top right, bottom right, bottom left
def order_points(pts):

    # Sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # Get the left-most and right-most points
    left_pts = x_sorted[:2, :]
    right_pts = x_sorted[2:, :]

    # Sort the left-most points based on their y-coordinates
    left_pts = left_pts[np.argsort(left_pts[:, 1]), :]
    top_left, bottom_left = left_pts

    # Sort the right-most points based on their y-coordinates
    right_pts = right_pts[np.argsort(right_pts[:, 1]), :]
    top_right, bottom_right = right_pts

    # Return the ordered coordinates
    return np.array([top_left, top_right, bottom_right, bottom_left])

# Define a function to resize images
def img_resize(image):

    height, width = image.shape[:2]
    min_size = 560
    # Determine if the smallest edge of an image is less than a threshold value
    if min(height, width) < min_size:
        # Calculate scaling
        scaling_factor = min_size / min(height, width)

        # New dimensions, maintain aspect ratio
        new_size = (int(width * scaling_factor), int(height * scaling_factor))

        # Isometric scaling of images
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return image

# Define a function to process images
def process_image(image):

    # Define global variables
    global stop_time
    global ocr_result

    pre_img = image.copy()
    # Resize the image
    pre_img = img_resize(pre_img)
    plt.figure(figsize=(18, 8))
    plt.gcf().canvas.set_window_title('Image Preprocessing')
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB))
    plt.title('original image')
    plt.axis('off')

    # Image enhancement
    origin = image_enhancement(pre_img)
    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(origin, cv2.COLOR_BGR2RGB))
    plt.title('image_enhancement')
    plt.axis('off')
    # Grayscale image conversion
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

    # Noise reduction through the use of Gaussian filtering
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    plt.subplot(2, 4, 3)
    plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    plt.title('Noise Removal')
    plt.axis('off')

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.title('sharpening')
    plt.axis('off')

    # Thresholding
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
    plt.title('Thresholding')
    plt.axis('off')


    # Morphological manipulation of expansion
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    dilation = cv2.dilate(binary, rect_kernel, iterations=1)
    # Morphological operations: expansion followed by corrosion (closed operations)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph= cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, rect_kernel)
    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(morph, cv2.COLOR_BGR2RGB))
    plt.title('morph')
    plt.axis('off')

    # Edge Detection
    edges = cv2.Canny(morph, 50, 150)
    # Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Define thresholds to control sensitivity to overlap
    min_black_ratio = 0.05
    max_black_ratio = 0.95
    iou_threshold = 0.1
    # Set a minimum area to avoid detecting small extraneous areas
    min_area = 300
    bounding_boxes = [cv2.boundingRect(ctr) for ctr in contours]
    num_boxes = len(bounding_boxes)
    # Filter outlines that are too small, thin, have few black pixels, and overlap too much
    final_contours = []
    for i in range(num_boxes):
        box1 = bounding_boxes[i]
        out = False
        x, y, w, h = box1
        # Filter out contours with areas less than a threshold value
        if w * h < min_area:
            continue
        # Calculate aspect ratios to exclude excessively flat or narrow silhouettes
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.2 or aspect_ratio > 20.0:
            continue
        # Calculate black pixel ratio
        black_pixel_ratio = compute_black_pixel_ratio(binary, contours[i])
        # If the proportion of black pixels is greater than the minimum threshold and less than the maximum threshold, it is considered a text block outline
        if black_pixel_ratio < min_black_ratio or black_pixel_ratio > max_black_ratio:
            continue
        for j in range(i + 1, num_boxes):
            box2 = bounding_boxes[j]
            iou = compute_iou(box1, box2)
            if iou > iou_threshold:
                out = True
                break
        if not out:
            final_contours.append(contours[i])

    # Contour Sorting
    need_newline, final_contours = sort_contours(final_contours)
    # Draw contours on the original image
    final = origin.copy()
    # Draw the detected contours on the original image
    cv2.drawContours(final, final_contours, -1, (0, 255, 0), 3)

    # Display the final image with all contours
    plt.subplot(2, 4, 7)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Contour image')
    plt.axis('off')
    block_time = time.time()
    plt.show()
    stop_time = stop_time + time.time() - block_time

    # Operate on individual blocks of text
    for i, contour in enumerate(final_contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Selected text blocks for the original noise-reduced image
        cropped = extract_contour_as_rectangular_image(contour, gray)

        # Gaussian blurring of text blocks
        denoised_text =  cv2.GaussianBlur(cropped, (5, 5), 0)

        # Sharpen the text block
        kernel = np.array([[0, -5, 0],
                           [-5, 21, -5],
                           [0, -5, 0]])
        sharpened_text = cv2.filter2D(denoised_text, -1, kernel)

        # Binarize blocks of text
        _, binary_text = cv2.threshold(sharpened_text, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        # Perform a Hough transform on a block of text
        rotate_text = detect_and_rotate_image(binary_text)

        '''
        # OCR with Tesseract
        config_1 = r'--oem 1 --psm 6 -l chi_sim+eng'
        result = pytesseract.image_to_string(rotate_text, config=config_1)
        if result :
                text = result.rstrip()
                if need_newline[i]:
                   sys.stdout.write("\n"+str(text)+" ")
                   sys.stdout.flush()
                   ocr_result += "\n" + str(text) + " "
                else:
                   sys.stdout.write(str(text)+" ")
                   sys.stdout.flush()
                   ocr_result += str(text) + " "
        '''
        # Make PaddleOCR return text
        result = ocr.ocr(rotate_text, cls=True)

        if result :
            for line in result:
                text = line[1][0].rstrip()
                if need_newline[i]:
                   sys.stdout.write("\n"+str(text)+" ")
                   sys.stdout.flush()
                   ocr_result += "\n" + str(text) + " "
                else:
                   sys.stdout.write(str(text)+" ")
                   sys.stdout.flush()
                   ocr_result += str(text) + " "


        # Display text blocks
        current = origin.copy()
        cv2.drawContours(current, [contour], -1, (0, 255, 0), 3)
        plt.figure(figsize=(9, 9))
        plt.gcf().canvas.set_window_title('Current text')
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 8])
        plt.subplot(gs[0])
        plt.imshow(cv2.cvtColor(rotate_text, cv2.COLOR_BGR2RGB))
        plt.title('Current text block')
        plt.axis('off')
        plt.subplot(gs[1])
        plt.imshow(cv2.cvtColor(current, cv2.COLOR_BGR2RGB))
        plt.title('Current text page')
        plt.axis('off')

        block_time = time.time()
        plt.show()
        stop_time = stop_time + time.time() - block_time

    # 2 empty lines
    print('\n')
    return final

# Define a function that calculates OCR performance
def calculate_accuracy(true_file, ocr_results):

    # Read and clean up real text, remove line breaks and split words by spaces
    with open(true_file, 'r', encoding='utf-8') as file:
        true_text = file.read().replace('\n', ' ')

    # OCR result removes line breaks and spaces, retains all characters
    ocr_results = ocr_results.replace('\n', '').replace(' ', '')

    origin_ocr_len = len(ocr_results)

    # Split real text into word lists by space
    true_words = true_text.split()

    # Record the number of successfully matched characters
    matched_chars = 0

    # Iterate over each word in the real text and split it into a sequence of characters and look it up in the OCR character sequence
    for word in true_words:
        # Convert words to character sequences
        word_as_chars = ''.join(word)
        # Maximum number of partially matched characters initialized to 0 to ensure integer
        max_partial_match_len = 0
        # Maximum matching OCR position initialized to -1
        max_match_position = -1

        # Check if the character sequence is present in the character result of the OCR
        if word_as_chars in ocr_results:
            # If an exact match, the number of matching characters plus the length of the character sequence
            matched_chars += len(word_as_chars)
            # Remove matched character sequences to avoid duplicate matches
            ocr_results = ocr_results.replace(word_as_chars, '', 1)
        else:
            # If there is no exact match, progressively check for partial matches in the OCR
            matcher = SequenceMatcher(None, word_as_chars, ocr_results)
            match = matcher.find_longest_match(0, len(word_as_chars), 0, len(ocr_results))

            # If the partial match found is greater than 1, update the maximum partial match value
            if match.size > 1:
                if match.size > max_partial_match_len:
                    max_partial_match_len = match.size
                    max_match_position = match.b

            # Add the longest partial match found to the number of matching characters only when it is greater than 1
            if max_partial_match_len > 1:
                matched_chars += max_partial_match_len
                # Remove the longest partially matched character sequence from the OCR result
                ocr_results = ocr_results[:max_match_position] + ocr_results[
                                                                 max_match_position + max_partial_match_len:]


    # Calculate the total number of characters in the real text
    total_true_chars = sum(len(word) for word in true_words)
    # Calculate the total number of characters in the OCR result
    total_ocr_chars = origin_ocr_len    # OCR字符总数

    # Calculate the precision (Precision) = Number of matched characters / Total number of OCR characters
    precision = matched_chars / total_ocr_chars if total_ocr_chars > 0 else 0

    # Calculate the recall (Recall) = Number of matched characters / Total number of real characters
    recall = matched_chars / total_true_chars if total_true_chars > 0 else 0

    # Calculate the F1 score (F1 Score) = 2 * (Precision * Recall) / (Precision + Recall)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    return precision, recall, f1_score

# Main process (program header)

# Load the image and text file
image = cv2.imread('image/img_2.png')
txt_path = 'image/img_2.txt'

image = img_resize(image)
warped = image.copy()
points = []
stop_time = 0
ocr_result = ''

# Display the image and bind the mouse click event
fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Bind the mouse click event
fig.canvas.mpl_connect('button_press_event', mouse_click)
plt.show()

# Loading the PaddleOCR tool
ocr = PaddleOCR(use_angle_cls=True, show_log=False, det_db_box_thresh=0.1, det_db_unclip_ratio=3)
start_time = time.time()
contour_image = process_image(warped)
# Calculate the total time taken
total_time = time.time() - start_time - stop_time
print(f"Total time: {total_time:.2f} seconds")

# Calculate OCR performance
precision, recall, f1_score = calculate_accuracy(txt_path, ocr_result)
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1_score * 100:.2f}%")

# Display the original image and the final image with all contours
plt.figure(figsize=(18, 9))
plt.gcf().canvas.set_window_title('OCR_RESULT')
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.title('Original image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title('Contour image')
plt.axis('off')
plt.show()

# Create a figure with 3 subplots, making sure all subplots have the same size
fig, axs = plt.subplots(3, 1, figsize=(16, 7))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Subplot 1: Contour Image (resized to fill the axes properly)
axs[0].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Contour Image')
axs[0].axis('off')  # Hide axes for the image

# Subplot 2: OCR Text Display (left-aligned text, fit within the plot)
axs[1].text(0.01, 0.5, ocr_result, fontsize=12, ha='left', va='center_baseline', wrap=True)
axs[1].set_title('OCR Result')
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].axis('off')  # Hide axes for the text display

# Subplot 3: Performance Metrics (Vertical bar plot with percentages)
metrics = ['Precision', 'Recall', 'F1 Score', 'Time (s)']
values = [precision * 100, recall * 100, f1_score * 100, total_time]  # Time is not scaled to percentage

# Adjusting bar width (setting bar width smaller to avoid filling the entire area)
bars = axs[2].bar(metrics, values, color=['blue', 'green', 'orange', 'red'], width=0.5)

# Add labels above the bars
for bar, metric, value in zip(bars, metrics, values):
    if metric == 'Time (s)':
        # Display time as it is, with 2 decimal places
        axs[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom')
    else:
        # Display percentage for other metrics, with 2 decimal places
        axs[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{value:.2f}%', ha='center', va='bottom')

# Adjust y-limits for a more balanced view
axs[2].set_ylim(0, max(values) * 1.2)  # Add some space above the tallest bar
axs[2].set_title('Performance Metrics')


# Adjust layout to make everything fit
plt.tight_layout()
plt.gcf().canvas.set_window_title('Paddle OCR_RESULT')
# Show the plot
plt.show()
