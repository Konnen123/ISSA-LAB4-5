from csv import excel

import cv2
import numpy as np

prev_right_bottom_x = 0
prev_right_top_x = 0

prev_left_bottom_x = 0
prev_left_top_x = 0

prev_right_bottom_y = 0
prev_right_top_y = 0

prev_left_bottom_y = 0
prev_left_top_y = 0

def get_trapezoid_array(width, height):
    upper_left = (width * 0.55, height * 0.76)
    upper_right = (width * 0.45, height * 0.76)
    bottom_left = (0, height)
    bottom_right = (width, height)

    return np.array([upper_left, upper_right, bottom_left, bottom_right], dtype=np.int32)

def get_road_only_frame(frame, trapezoid_points):
    # Draw the trapezoid on a black frame
    black_frame = frame.copy()
    black_frame.fill(0)

    # Fill the trapezoid area with white color
    black_frame_with_trapezoid = cv2.fillConvexPoly(black_frame, trapezoid_points, (255, 255, 255))

    # Create a mask with the trapezoid area by performing a bitwise AND between the original frame and the black frame with trapezoid
    # Because only the trapezoid area will be white, the part that will not be inside the trapezoid will be black because of the AND operationz
    return cv2.bitwise_and(frame, black_frame_with_trapezoid)

def get_top_down_blurred_frame(frame, trapezoid_points):
    # We want to detect the lane markings on the street, and distant markings are just as important as closer ones

    # Get the trapezoid coordinates
    trapezoid_bounds = np.float32(trapezoid_points)

    # Get the frame coordinates
    frame_bounds = np.float32(np.array([(frame.shape[1], 0), (0, 0), (0, frame.shape[0]),
                                        (frame.shape[1], frame.shape[0])], dtype=np.int32))

    # Get the transformation matrix that will help us to get the top-down stretched view
    top_down_matrix = cv2.getPerspectiveTransform(trapezoid_bounds, frame_bounds)

    # Using the transformation matrix, we can get the top-down view of the trapezoid
    top_down_frame = cv2.warpPerspective(frame, top_down_matrix,
                                         (frame.shape[1], frame.shape[0]))

    # Step 6: Blur the top-down view
    # To blur we need to make each pixel to be the average of its neighbors
    # If we use more neighbors, the blur will be more intense
    # Use a matrix of size 3x3 that is also called a kernel
    # The kernel must be odd, so we can have a center pixel
    return cv2.blur(top_down_frame, (3, 3))

def get_sobel_frame_binarized():

    # We have a matrix called filter that is used to detect the edges
    # Move the filter over each pixel so each one is in center
    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])

    sobel_horizontal = np.transpose(sobel_vertical)

    # Convert the frame to float32 because the filter will not work with uint8
    top_down_blurred_frame_as_float = np.float32(top_down_blurred_frame)

    # Apply the filter to the frame
    # The -1 parameter specifies that depth of the output image will be the same as the input image
    sobel_vertical_frame = cv2.filter2D(top_down_blurred_frame_as_float, -1, sobel_vertical)
    sobel_horizontal_frame = cv2.filter2D(top_down_blurred_frame_as_float, -1, sobel_horizontal)

    # To combine the two frames, we need to square each pixel and then sum them
    # matrix3 = sqrt(matrix1^2 + matrix2^2)
    sobel_frame = np.sqrt(np.square(sobel_vertical_frame) + np.square(sobel_horizontal_frame))

    # Convert the frame back to uint8 with convertScaleAbs because the final image needs to be in uint8
    # This method handles scaling and absolute value conversion
    sobel_frame_final = cv2.convertScaleAbs(sobel_frame)

    # Step 8: Binarize the image
    # Either convert each pixel to 0 or 255 (max value)
    # The threshold is the value that will be used to convert the pixel
    _, sobel_frame_final = cv2.threshold(sobel_frame_final, 180, 255, cv2.THRESH_BINARY)
    #sobel_frame_final = np.where(sobel_frame_final > 160, 255, 0).astype(np.uint8)

    return sobel_frame_final

def update_prev_values(left_top_x, left_bottom_x, right_top_x, right_bottom_x, left_top_y, left_bottom_y, right_top_y, right_bottom_y):
    global prev_left_top_x
    global prev_left_bottom_x
    global prev_right_top_x
    global prev_right_bottom_x
    global prev_left_top_y
    global prev_left_bottom_y
    global prev_right_top_y
    global prev_right_bottom_y

    prev_left_top_x = left_top_x
    prev_left_bottom_x = left_bottom_x
    prev_right_top_x = right_top_x
    prev_right_bottom_x = right_bottom_x
    prev_left_top_y = left_top_y
    prev_left_bottom_y = left_bottom_y
    prev_right_top_y = right_top_y
    prev_right_bottom_y = right_bottom_y

def get_indices(frame, trapezoid_points):
    trapezoid_bounds = np.float32(trapezoid_points)
    blank_frame_bounds = np.float32(np.array([(frame.shape[1], 0), (0, 0), (0, frame.shape[0]),
                                              (frame.shape[1], frame.shape[0])], dtype=np.int32))

    magic_matrix = cv2.getPerspectiveTransform(blank_frame_bounds, trapezoid_bounds)

    frame_with_line = cv2.warpPerspective(frame, magic_matrix,
                                               (frame.shape[1], frame.shape[0]))

    return np.argwhere(frame_with_line >= 1)


if __name__ == '__main__':
    cam = cv2.VideoCapture('Lane_Detection_Test_Video_01.mp4')

    while True:
        ret, frame = cam.read()

        # ret (bool): Return code of the `read` operation. Did we get an image or not?
        #            (if not maybe the camera is not detected/connected etc.)
        # frame (array): The actual frame as an array.
        #                Height x Width x 3 (3 colors, BGR) if color image.
        #                Height x Width if Grayscale
        #                Each element is 0-255.
        #                You can slice it, reassign elements to change pixels, etc.

        if ret is False:
            break

        # Step 1: Shrink the frame
        shrink_frame = cv2.resize(frame, (frame.shape[1] // 5, frame.shape[0] // 5))

        # Step 2: Convert to grayscale
        gray_frame = cv2.cvtColor(shrink_frame, cv2.COLOR_BGR2GRAY)

        # Step 3: Get trapezoid points for the road(int32 array)
        trapezoid_points = get_trapezoid_array(gray_frame.shape[1], gray_frame.shape[0])

        # Using the trapezoid points, we can create a mask to get the road only
        road_only_frame = get_road_only_frame(gray_frame, trapezoid_points)

        # Step 5: Stretch the trapezoid to the whole frame
        top_down_blurred_frame = get_top_down_blurred_frame(road_only_frame, trapezoid_points)

        # Step 6: Do the edge detection using the Sobel filter
        sobel_frame_binarized = get_sobel_frame_binarized()

        # Step 8: Get the coordinates of street markings on each side of the road
        # Each half will have its own line
        sobel_frame_binarized_copy = sobel_frame_binarized.copy()

        # Set the first 5% of columns (from the left) in all rows to 0
        sobel_frame_binarized_copy[:, :int(sobel_frame_binarized_copy.shape[1] * 0.05)] = 0
        # Sets the last 15% of columns (from the right) in all rows to zero.
        sobel_frame_binarized_copy[:, int(sobel_frame_binarized_copy.shape[1] * 0.85):] = 0
        # Sets the first 5% of rows (from the top) in all columns to zero.
        sobel_frame_binarized_copy[:, :int(sobel_frame_binarized_copy.shape[0] * 0.05)] = 0

        # Get the left and right halves of the image
        left_sobel_frame = sobel_frame_binarized_copy[:, :int(sobel_frame_binarized_copy.shape[1] * 0.5)]
        right_sobel_frame = sobel_frame_binarized_copy[:, int(sobel_frame_binarized_copy.shape[1] * 0.5):]

        # Get the coordinates (x,y) of each white point using argwhere
        # np.argwhere will give us the coordinates in the form of (y,x)
        left_indices = np.argwhere(left_sobel_frame >= 1)
        left_x_indices = left_indices[:, 1]
        left_y_indices = left_indices[:, 0]

        right_indices = np.argwhere(right_sobel_frame >= 1)
        right_x_indices = right_indices[:, 1] + int(sobel_frame_binarized_copy.shape[1] * 0.5)
        right_y_indices = right_indices[:, 0]

        # Step 10: find the lines that detect the edges of the lane
        left_line = None
        right_line = None
        try:
            left_line = np.polynomial.polynomial.polyfit(left_x_indices, left_y_indices, deg=1)
            right_line = np.polynomial.polynomial.polyfit(right_x_indices, right_y_indices, deg=1)
        except TypeError:
            pass

        left_top_y = 0
        left_bottom_y = left_sobel_frame.shape[0]

        left_top_x = pow(10, 9)
        left_bottom_x = pow(10, 9)
        if left_line is not None:
            left_top_x = int((left_top_y - left_line[0]) / left_line[1])
            left_bottom_x = int((left_bottom_y - left_line[0]) / left_line[1])

        right_top_y = 0
        right_bottom_y = right_sobel_frame.shape[0]

        right_top_x = pow(10, 9)
        right_bottom_x = pow(10, 9)
        if right_line is not None:
            right_top_x = int((right_top_y - right_line[0]) / right_line[1])
            right_bottom_x = int((right_bottom_y - right_line[0]) / right_line[1])

        minimum = pow(10, -8)
        maximum = pow(10, 8)
        if right_top_x < minimum or right_top_x > maximum:
            right_top_x = prev_right_top_x

        if right_bottom_x < minimum or right_bottom_x > maximum:
            right_bottom_x = prev_right_bottom_x

        if left_top_x < minimum or left_top_x > maximum:
            left_top_x = prev_left_top_x

        if left_bottom_x < minimum or left_bottom_x > maximum:
            left_bottom_x = prev_left_bottom_x

        update_prev_values(left_top_x, left_bottom_x, right_top_x, right_bottom_x, left_top_y, left_bottom_y, right_top_y, right_bottom_y)

        left_top = int(left_top_x), int(left_top_y)
        left_bottom = int(left_bottom_x), int(left_bottom_y)

        right_top = int(right_top_x), int(right_top_y)
        right_bottom = int(right_bottom_x), int(right_bottom_y)

        cv2.line(sobel_frame_binarized_copy, left_top, left_bottom, (200, 0, 0), 5)
        cv2.line(sobel_frame_binarized_copy, right_top, right_bottom, (100, 0, 0), 5)

        blank_frame_left = np.zeros_like(shrink_frame)
        cv2.line(blank_frame_left, left_top, left_bottom, (255, 0, 0), 3)

        blank_frame_right = np.zeros_like(shrink_frame)
        cv2.line(blank_frame_right, right_top, right_bottom, (255, 0, 0), 3)

        left_indices = get_indices(blank_frame_left, trapezoid_points)
        right_indices = get_indices(blank_frame_right, trapezoid_points)

        shrink_frame_copy = shrink_frame.copy()
        shrink_frame_copy[left_indices[:, 0], left_indices[:, 1]] = [50, 50, 250]
        shrink_frame_copy[right_indices[:, 0], right_indices[:, 1]] = [50, 250, 50]


        cv2.imshow('Original', shrink_frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()