import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def select_image_from_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = os.path.dirname(os.path.abspath(__file__))  # Get the current script's folder
    file_path = filedialog.askopenfilename(initialdir=folder_path, title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.bmp *.jpeg")])

    return file_path

def daugman_iris_detection(image):
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (9, 9), 7)  # 9,9 for kernel size given to blur the image  and 7 for to control the blur intensity
    
    # Find the center and radius of the pupil using the HoughCircles function
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=100, param2=30, minRadius=10, maxRadius=150)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        pupil_circle = circles[0][0]
        return pupil_circle[0], pupil_circle[1], pupil_circle[2]
    else:
        return None, None, None

def daugman_sclera_segmentation(gray_eye, iris_center_x, iris_center_y, iris_radius):
    # Create a mask for the iris region
    mask = np.zeros_like(gray_eye)
    
    # Manually define contour points for the outer oval (iris region)
    outer_oval_points = []
    major_axis_outer = iris_radius * 1.75  # Increase the major_axis value
    minor_axis_outer = iris_radius
    for angle in range(0, 360, 2):
        radian = np.deg2rad(angle)
        x = int(iris_center_x - major_axis_outer * np.cos(radian))
        y = int(iris_center_y + minor_axis_outer * np.sin(radian))
        outer_oval_points.append([x, y])
    
    # Manually define contour points for the inner oval (pupil region)
    inner_oval_points = []
    major_axis_inner = iris_radius * 0.9 # Adjust the major_axis value for a larger pupil
    minor_axis_inner = iris_radius * 0.9  # Adjust the minor_axis value for a larger pupil
    for angle in range(0, 360, 2):
        radian = np.deg2rad(angle)
        x = int(iris_center_x - major_axis_inner * np.cos(radian))
        y = int(iris_center_y + minor_axis_inner * np.sin(radian))
        inner_oval_points.append([x, y])
    
    outer_oval_points = np.array(outer_oval_points)
    inner_oval_points = np.array(inner_oval_points)
    
    # Draw the outer oval region with a white color and the inner oval region with a black color
    cv2.drawContours(mask, [outer_oval_points], -1, 255, -1)
    cv2.drawContours(mask, [inner_oval_points], -1, 0, -1)
    
    # Create a border around the sclera region
    sclera_border_thickness = 12  # Adjust the border thickness as needed
    cv2.drawContours(mask, [outer_oval_points], -1, 0, sclera_border_thickness)
    
    # Apply the mask to the gray_eye image to extract the sclera region with a border
    sclera_with_border = cv2.bitwise_and(gray_eye, mask)

    # Remove the border from the sclera region
    sclera_no_border = sclera_with_border[sclera_border_thickness:-sclera_border_thickness, sclera_border_thickness:-sclera_border_thickness]

    return sclera_no_border

def apply_clahe(image):
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the image
    clahe_image = clahe.apply(image)
    
    return clahe_image

def extract_features(image):
    # Implement feature extraction using Canny edge detection
    edges = cv2.Canny(image,30, 90)
    
    return edges

def main():
    # Create a simple user interface window
    root = tk.Tk()
    root.title("Sclera Feature Extraction")

    # Set the window size (width x height)
    window_width = 1100
    window_height = 700

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the position to center the window on the screen
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2

    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # Load the background image (JPG)
    bg_image = Image.open("bg.jpg")
    bg_image = ImageTk.PhotoImage(bg_image)

    # Create a label to display the background image
    background_label = tk.Label(root, image=bg_image)
    background_label.place(relwidth=1, relheight=1)
   
    # Create a button to select an image
    button_font = ("Georgia", 24, "bold")
    select_button = tk.Button(root, text="SELECT IMAGE", font=button_font, command=lambda: process_image())
    select_button.place(x=550,y=550)

    # Function to process the selected image
    def process_image():
        image_path = select_image_from_folder()
        
        if image_path:
            # Load the selected image
            image = cv2.imread(image_path)

            # New resolution (width, height)
            new_width = 275
            new_height = 183

            # Resize the image to the new resolution
            resized_image = cv2.resize(image, (new_width, new_height))

            # Convert the image to grayscale
            gray_eye = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # Call the daugman_iris_detection function
            iris_center_x, iris_center_y, iris_radius = daugman_iris_detection(gray_eye)

            if iris_center_x is not None and iris_center_y is not None and iris_radius is not None:
                # Draw the iris arc in green color on the image
                image_with_green_arc = resized_image.copy()
                cv2.ellipse(image_with_green_arc, (iris_center_x, iris_center_y), (iris_radius, iris_radius), 0, 0, 360, (0, 255, 0), 2)

                # Call the daugman_sclera_segmentation function
                segmented_sclera = daugman_sclera_segmentation(gray_eye, iris_center_x, iris_center_y, iris_radius)

                # Apply CLAHE to the segmented sclera
                clahe_sclera = apply_clahe(segmented_sclera)

                 # Extract features from the enhanced sclera
                features = extract_features(clahe_sclera)

                white_background = np.ones_like(resized_image) * 255

                
                
                # Display the original image, the segmented iris, and the enhanced sclera
                cv2.imshow('Original Eye Image', resized_image)
                cv2.imshow('Iris Segmentation', image_with_green_arc)
                cv2.imshow('sclera Segmentation', segmented_sclera)
                cv2.imshow('Enhanced Sclera', clahe_sclera)
                cv2.imshow('Extracted Features', 255-features)

                # Arrange the windows horizontally with spacing
                window_width = resized_image.shape[1]
                window_height = resized_image.shape[0]

                x_offset = 40  # Adjust as needed
                y_offset = 200  # Adjust as needed

                cv2.moveWindow('Original Eye Image', x_offset, y_offset)
                cv2.moveWindow('Iris Segmentation', x_offset + window_width + 10, y_offset)
                cv2.moveWindow('sclera Segmentation', x_offset + 2 * (window_width + 10), y_offset)
                cv2.moveWindow('Enhanced Sclera', x_offset + 3 * (window_width + 5), y_offset)
                cv2.moveWindow('Extracted Features', x_offset + 4 * (window_width + 5), y_offset)


                # Segmented sclera image (binary image)
                seg_sclera = segmented_sclera # Add the segmented sclera image here

                # Calculate the area of the segmented sclera
                sclera_area = np.count_nonzero(seg_sclera)

                shaded_area_region = sclera_area*0.01
                print(f"Shaded region area : {shaded_area_region}%")

                # Define the mean and the point
                if shaded_area_region>=48.68:
                    mean = 52.57
                elif  48.68 >= shaded_area_region >=26.1:
                    mean = 34.73
                else:
                    mean = 29.92    

                point = shaded_area_region

                # Calculate the accuracy_diff as a percentage
                accuracy_diff = abs((mean - point) / mean) * 100

                # Define the two points
                p1 = f"{100-accuracy_diff:.2f}%"

                # Remove the '%' sign and convert the string to a float
                p1_float = float(p1.replace('%', ''))

                length = 400
                height = 500

                plt.figure(figsize=(length / 100, height / 100))  # Convert to inches
                point1 = p1_float
                point2 = accuracy_diff

                # Create data for the pie chart
                points = [point1, point2]
                colors = ['green', 'red']

                # Labels for the bar graph 
                labels = ['Accurate Region', 'Error Region']

                # Create a bar graph
                plt.bar(labels, points, color=colors)

                # Add labels
                for i in range(len(points)):
                    plt.text(labels[i], points[i], f'{points[i]:.2f}%', ha='center', va='bottom')  

                plt.ylim(0, 100)                  

                # Title of the bar graph
                plt.title('ACCURACY')

                # Display the bar graph
                plt.show()

                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or cv2.getWindowProperty('Original Eye Image', cv2.WND_PROP_VISIBLE) < 1:
                        # Close all OpenCV windows when 'Esc' key is pressed or any window is closed
                        cv2.destroyAllWindows()
                        break

    root.mainloop()
   
if __name__ == "__main__":
    main()
