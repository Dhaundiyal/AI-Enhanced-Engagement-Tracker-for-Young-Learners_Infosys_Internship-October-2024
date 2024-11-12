# AI-Enhanced-Engagement-Tracker-for-Young-Learners_Infosys_Internship-October-2024
## Annotations
### data_segregate.py:
This script automates the process of segregating images based on the availability of their corresponding label files. It ensures that images with matching labels are organized into one folder, while unmatched images are moved to another.
### label_manipylate.py:
This script modifies class IDs in label files, useful for tasks involving object detection or classification annotations. It searches through label files in a specified directory, identifies lines with a specified class ID, and replaces them with a new class ID.
### labelimg.py:
This script overlays bounding boxes on images based on corresponding label files, helping visualize object detection annotations.
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
