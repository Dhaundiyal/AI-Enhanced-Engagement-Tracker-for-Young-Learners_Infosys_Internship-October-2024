# AI-Enhanced-Engagement-Tracker-for-Young-Learners_Infosys_Internship-October-2024
## Annotations
Here we use the labelimg library in python to perform the following tasks.

_The images, text and output files are in the Images folder_
### Dependencies:
Python 3.x

os and shutil modules (built-in)

cv2

### data_segregate.py:
This script automates the process of segregating images based on the availability of their corresponding label files. It ensures that images with matching labels are organized into one folder, while unmatched images are moved to another.
### label_manipulate.py:
This script modifies class IDs in label files, useful for tasks involving object detection or classification annotations. It searches through label files in a specified directory, identifies lines with a specified class ID, and replaces them with a new class ID.

![image](https://github.com/user-attachments/assets/b4b218a4-b4c2-4faa-89a4-e46bf33a5114)
### labelimg.py:
This script overlays bounding boxes on images based on corresponding label files, helping visualize object detection annotations.

 ![image](https://github.com/user-attachments/assets/6389b6a5-a752-41eb-a7c9-5fcf68e17eef)
 ![image](https://github.com/user-attachments/assets/1c60ffe9-4540-4457-a07e-c7cdcacc5510)
## Image Processing

### Dependencies:
Python 3.x

cv2

numpy

matplotlib

### Input images
![image](https://github.com/user-attachments/assets/892022a7-4317-4e97-99e0-eb88cf4811d5)
![image](https://github.com/user-attachments/assets/d744c855-e46a-4ba5-b5ad-d00324fb1c30)

In image processing we perform multiple oprerations on the two given input images and see how different operations perform.

### bgr2gray.py:
This script converts a color image to grayscale and saves the result

![image](https://github.com/user-attachments/assets/6e4b4534-039f-4c72-8e4d-2f9038635448)
### dil_ero.py
This script demonstrates the use of basic morphological operations, dilation and erosion, on a grayscale image. These operations are often used in image preprocessing tasks such as noise removal, image enhancement, and shape extraction.

![image](https://github.com/user-attachments/assets/53cc438b-5fc9-4114-84b6-b2ace2cf6bbd)

### histogram_equalization.py
This script performs histogram equalization on a grayscale image to enhance its contrast. Histogram equalization improves the visibility of features in images with poor contrast, often used in image preprocessing tasks.

![image](https://github.com/user-attachments/assets/698f801e-4b01-40df-aa57-a2714d030caa)

### img_blur.py
This script applies a Gaussian blur to an image, which is commonly used for noise reduction or creating artistic effects by smoothing the image. The blur effect softens the details of the image, making it appear less sharp.

![image](https://github.com/user-attachments/assets/6066a14d-31c9-4153-afe2-7d7b87490968)

### img_contour.py
This script detects and draws contours on a grayscale image. Contour detection is useful in various computer vision tasks, such as object detection, shape analysis, and image segmentation.

![image](https://github.com/user-attachments/assets/4ab30fcb-e2aa-4d92-b3e1-7644cb23e7b5)

### img_crop.py
This script crops a portion of the input image based on specified pixel coordinates. Cropping is useful for focusing on a region of interest (ROI) or removing unwanted parts of an image.

![image](https://github.com/user-attachments/assets/92191c1d-262b-41ce-9eba-34cafa56a1d1)

### img_edge.py
This script applies the Canny edge detection algorithm to a grayscale image, which is used to detect edges by finding areas of rapid intensity change. It is widely used in image processing tasks like feature detection and object recognition.

![image](https://github.com/user-attachments/assets/70ba5941-9c4d-4315-9a99-5a56424092dc)

### img_hsv.py
This script converts a color image from the BGR (Blue, Green, Red) color space to the HSV (Hue, Saturation, Value) color space.

![image](https://github.com/user-attachments/assets/69316042-c23f-4b95-8447-9d6b9b607131)

### img_noiserm.py
This script demonstrates the use of morphological operations on a grayscale image to process and refine image structures. It applies opening and closing operations to remove noise and fill gaps in the image, respectively.
### img_resize.py
This script resizes an input image to a specified dimension. Image resizing is useful when you need to adjust image size for display purposes, machine learning models, or storage optimization.

![image](https://github.com/user-attachments/assets/1dab0e4d-513c-4c9b-a769-9848d5bbb01e)

### img_rotate.py
This script rotates an image by a specified angle (90 degrees in this case) around its center.

![image](https://github.com/user-attachments/assets/bcbac200-bf37-4ca8-9785-4d5a1ed94417)

### img_stack.py
This script concatenates two images both horizontally and vertically. Image concatenation is useful when you want to combine multiple images into a single image for comparison, visualization, or collage creation.

![image](https://github.com/user-attachments/assets/d50d4057-cb94-4895-98f1-f10a8f5f0bf4)
![image](https://github.com/user-attachments/assets/b46ae7a3-6176-4e8a-90b7-5ba027484c6b)

### morphological_transformation.py
This script demonstrates the use of morphological operations (opening and closing) on a grayscale image. These operations are used to process image structures, such as removing noise or filling gaps in binary images or objects within an image.

![image](https://github.com/user-attachments/assets/9ab0156f-997f-4dcf-9660-80514c0aae47)

### template.py
This script uses template matching to find a template image within a larger image. It draws a bounding box around the detected template in the source image. 

## Video Processing

### Dependencies:
Python 3.x

cv2

In video processing we see how to capture a video using webcam or a given video and perform the operations mentioned below.

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

### Dependencies:
Python 3.x

cv2

face_recognition

pandas

numpy

xlrd

openpyxl

dlib

imutils

In this project we build an attendace system by using face recognition and saving necessary data to evaluate the attendace, the attentive score, screenshots with date and time etc.

_The .dat file used is in the folder shape_predictor_68_face_landmark. Specify the path while running the code._ 

### Input image

![image](https://github.com/user-attachments/assets/17d7e8ef-8bfd-4342-9d8e-e55f8ff453f3)

### 1_face_recog
This script uses the face_recognition library to detect and identify a specific individual from a live webcam feed.
### 2_attendace_save
This script implements a real-time face recognition system using a webcam. It detects and identifies a known face (e.g., 'Manas') and logs the attendance details (Name, Date, and Time) into an Excel file.

![image](https://github.com/user-attachments/assets/ec5a9b52-7e83-404d-8ab9-5f8488384930)

### 3_attendace_save_2
This script enhances a real-time face recognition system by adding time-based logic to manage attendance logging.

![image](https://github.com/user-attachments/assets/2cb59fd9-a2d6-42fd-aad5-1352716b1af3)

### 4_excel_dt
This script demonstrates a real-time face recognition system using face_recognition and OpenCV. It captures frames from a live video stream, recognizes a pre-defined individual, and logs recognition details into an Excel file along with screenshots.

![image](https://github.com/user-attachments/assets/8b8634b6-32c7-49ca-9f8d-dfb7457fca38)
![image](https://github.com/user-attachments/assets/d2ce77c7-bb62-41d8-9a6f-97824d6bc3aa)

### 5_excel_sc_dt
This script implements a real-time face recognition system that captures frames from a live video stream, detects a specific individual, and logs the recognition data. Each recognition event is recorded with a timestamped screenshot, ensuring clear context for every entry.

![image](https://github.com/user-attachments/assets/0a64b908-b61f-4872-941d-6c7873f6b78f)
![image](https://github.com/user-attachments/assets/1b4a82dc-aa39-497c-921b-336d6734a70b)

### 6_landmark
This script implements a real-time face recognition system that not only identifies a specific individual but also evaluates their attentiveness using head pose estimation. It logs recognized face data with timestamped screenshots and attentiveness status.

![image](https://github.com/user-attachments/assets/19548e8f-434a-4c75-a408-b0cc3cd7a8ac)
![image](https://github.com/user-attachments/assets/28512c12-d311-4103-aaac-7e8b10e0c9ce)

### 7_atten_score
This script extends a real-time face recognition system by introducing an attention scoring mechanism. It not only recognizes a specific individual but also computes their attentiveness score based on head pose estimation. The system logs recognized face data, including attention scores, and saves them along with screenshots.

![image](https://github.com/user-attachments/assets/8f631acd-d056-4d87-adc3-bbbedec9cefc)
![image](https://github.com/user-attachments/assets/5dd4b44e-ae4c-471a-8bac-81b2ebd7313f)

### 8_avg_atten_score
This script enhances the above script by adding the average attention scores for the users.

![image](https://github.com/user-attachments/assets/7d0ad457-6f4c-4dec-bc64-879ac6e72352)
![image](https://github.com/user-attachments/assets/e5af4ce1-34cd-48de-8d46-2ca93c157c31)

