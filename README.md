# Number Plate Detection using YOLO and OCR with Tesseract

This repository contains a script for detecting number plates in images using YOLO (You Only Look Once) and performing Optical Character Recognition (OCR) with Tesseract.

## Overview

The script utilizes the YOLO model for object detection and Tesseract for OCR. It performs the following steps:

- Convert the image to YOLO format.
- Get predictions from the YOLO model.
- Filter detections based on confidence and probability score.
- Apply Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes.
- Perform OCR on the detected license plate region using Tesseract.
- Visualize the results with bounding boxes and extracted text.
- Both OCR and PYTESSERACT results are shown.

## Dependencies

Make sure to install the required dependencies before running the script:

```bash
!git clone https://github.com/ultralytics/yolov5
```

```bash
!pip install -r ./yolov5/requirements.txt
```

## Usage
- Ensure that the YOLO model file (best.onnx) is present in the specified path (../input/cheemdur/best.onnx).
- Run the script with an image file path as an input:
```bash
python number_plate_detection.py path/to/your/image.jpg
```
## Customization
- Adjust confidence thresholds for YOLO detection and OCR based on your specific use case.
- Tune OCR parameters for better text recognition.
- Tune the script for potential modifications or improvements.

## Acknowledgments

- Optical Character Recognition (OCR) functionality is powered by the [easyocr](https://github.com/JaidedAI/EasyOCR) library. EasyOCR provides an easy-to-use interface for OCR tasks.

- Special thanks to the developers and contributors of the open-source libraries and frameworks used in this project, including OpenCV, NumPy, TensorFlow, pytesseract, Plotly, and scikit-image.

- The script is inspired by and adapted from various sources in the computer vision and OCR communities. We appreciate the collective efforts of the community in advancing these technologies.

If you find the work of any specific individual or organization particularly helpful, consider mentioning them by name and providing a link to their relevant repositories or resources.

## License
This project is licensed under the MIT License.

## Sample Result and Output
Sample images output :

![Output1](https://github.com/RidwanulHaque111/Vehicle-Number-Plate-Detection-CSE-472-Machine-Learning-Project-/blob/61a695e488754aaba2e5a6e8509fa799e3e59d40/output%20example/output1.JPG)

![Output2](https://github.com/RidwanulHaque111/Vehicle-Number-Plate-Detection-CSE-472-Machine-Learning-Project-/blob/61a695e488754aaba2e5a6e8509fa799e3e59d40/output%20example/output2.JPG)

![Output3](https://github.com/RidwanulHaque111/Vehicle-Number-Plate-Detection-CSE-472-Machine-Learning-Project-/blob/61a695e488754aaba2e5a6e8509fa799e3e59d40/output%20example/output3.JPG)
