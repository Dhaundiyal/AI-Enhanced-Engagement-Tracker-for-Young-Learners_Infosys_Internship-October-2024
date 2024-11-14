# AI-Enhanced-Engagement-Tracker-for-Young-Learners_Infosys_Internship-October-2024
## Annotations
Here we use the labelimg library in python to perform the following tasks.
### data_segregate.py:
This script automates the process of segregating images based on the availability of their corresponding label files. It ensures that images with matching labels are organized into one folder, while unmatched images are moved to another.
### label_manipylate.py:
This script modifies class IDs in label files, useful for tasks involving object detection or classification annotations. It searches through label files in a specified directory, identifies lines with a specified class ID, and replaces them with a new class ID.
### labelimg.py:
This script overlays bounding boxes on images based on corresponding label files, helping visualize object detection annotations.
 ![image](https://github.com/user-attachments/assets/6389b6a5-a752-41eb-a7c9-5fcf68e17eef)
 ![image](https://github.com/user-attachments/assets/1c60ffe9-4540-4457-a07e-c7cdcacc5510)
## Image Processing
### bgr2gray.py:
This script converts a color image to grayscale and saves the result
### dil_ero.py
This script demonstrates the use of basic morphological operations, dilation and erosion, on a grayscale image. These operations are often used in image preprocessing tasks such as noise removal, image enhancement, and shape extraction.
### histogram_equalization.py
This script performs histogram equalization on a grayscale image to enhance its contrast. Histogram equalization improves the visibility of features in images with poor contrast, often used in image preprocessing tasks.
### img_blur.py
This script applies a Gaussian blur to an image, which is commonly used for noise reduction or creating artistic effects by smoothing the image. The blur effect softens the details of the image, making it appear less sharp.
### img_contour.py
This script detects and draws contours on a grayscale image. Contour detection is useful in various computer vision tasks, such as object detection, shape analysis, and image segmentation.
### img_crop.py
This script crops a portion of the input image based on specified pixel coordinates. Cropping is useful for focusing on a region of interest (ROI) or removing unwanted parts of an image.
### img_edge.py
This script applies the Canny edge detection algorithm to a grayscale image, which is used to detect edges by finding areas of rapid intensity change. It is widely used in image processing tasks like feature detection and object recognition.
### img_hsv.py
This script converts a color image from the BGR (Blue, Green, Red) color space to the HSV (Hue, Saturation, Value) color space.
### img_noiserm.py
This script demonstrates the use of morphological operations on a grayscale image to process and refine image structures. It applies opening and closing operations to remove noise and fill gaps in the image, respectively.
### img_resize.py
This script resizes an input image to a specified dimension. Image resizing is useful when you need to adjust image size for display purposes, machine learning models, or storage optimization.
### img_rotate.py
This script rotates an image by a specified angle (90 degrees in this case) around its center.
### img_stack.py
This script concatenates two images both horizontally and vertically. Image concatenation is useful when you want to combine multiple images into a single image for comparison, visualization, or collage creation.
### morphological_transformation.py
This script demonstrates the use of morphological operations (opening and closing) on a grayscale image. These operations are used to process image structures, such as removing noise or filling gaps in binary images or objects within an image.
### template.py
This script uses template matching to find a template image within a larger image. It draws a bounding box around the detected template in the source image. 
## Video Processing
### multivid.py
This script loads and displays images from a folder one by one, printing their dimensions. It works by reading all files in a specified directory, attempting to load each file as an image, and showing it in a window.
### vid_fps.py
This script captures video from your webcam, displays it in a window, and saves the video to a file while displaying the frames per second (FPS) in real-time.
### vid_save.py
This script captures video from your webcam and saves it as a .avi file. It also displays the live video feed in a window. The video is saved in XVID format with a resolution of 640x480 at 20 frames per second. Press the 'q' key to stop the recording and close the window.
### vid_stack.py
This script reads two video files (video1.avi and video2.avi), resizes their frames, concatenates them horizontally, and displays the resulting video in real-time. It stops when either of the videos ends or if the user presses the 'q' key.
### vid_stream.py
This script captures live video from the default webcam and displays it in a window. The feed continues until the user presses the 'q' key to stop it.
## Face Recognition
### 1_face_recog
This script uses the face_recognition library to detect and identify a specific individual from a live webcam feed.
### 2_attendace_save
This script implements a real-time face recognition system using a webcam. It detects and identifies a known face (e.g., 'Manas') and logs the attendance details (Name, Date, and Time) into an Excel file.
### 3_attendace_save_2
This script enhances a real-time face recognition system by adding time-based logic to manage attendance logging.
### 4_excel_dt
This script demonstrates a real-time face recognition system using face_recognition and OpenCV. It captures frames from a live video stream, recognizes a pre-defined individual, and logs recognition details into an Excel file along with screenshots.
### 5_excel_sc_dt
This script implements a real-time face recognition system that captures frames from a live video stream, detects a specific individual, and logs the recognition data. Each recognition event is recorded with a timestamped screenshot, ensuring clear context for every entry.
### 6_landmark
This script implements a real-time face recognition system that not only identifies a specific individual but also evaluates their attentiveness using head pose estimation. It logs recognized face data with timestamped screenshots and attentiveness status.
### 7_atten_score
This script extends a real-time face recognition system by introducing an attention scoring mechanism. It not only recognizes a specific individual but also computes their attentiveness score based on head pose estimation. The system logs recognized face data, including attention scores, and saves them along with screenshots.
### 8_avg_atten_score
This script enhances the above script by adding the average attention scores for the users.
