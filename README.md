
# Text Detection and Recognition System

This project implements text detection and Optical Character Recognition (OCR) for extracting text from images, specifically handling Simplified Chinese and English. It compares the performance of two OCR engines: PaddleOCR and Tesseract. The project leverages OpenCV for image preprocessing and contour detection to accurately detect and extract text blocks from images.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Evaluation](#performance-evaluation)
- [Results](#results)
- [References](#references)

## Overview
Optical Character Recognition (OCR) technology is key to converting images containing text into machine-readable data. This project evaluates the performance of two OCR tools, PaddleOCR and Tesseract, through the following steps:
1. **Image Preprocessing**: Enhance image quality for better text detection.
2. **Text Detection**: Use edge and contour detection to locate text blocks.
3. **Text Recognition**: Apply OCR engines (PaddleOCR and Tesseract) to extract text from the detected blocks.
4. **Performance Metrics**: Measure precision, recall, and F1-score of the OCR results.

## Features
- **Text Detection**: Uses Canny edge detection and contour detection to identify text regions.
- **PaddleOCR and Tesseract**: Compares the performance of these two OCR engines on various types of text.
- **Performance Metrics**: Precision, recall, and F1-score are calculated to assess the quality of text recognition.
- **Text Preprocessing**: Includes techniques like image scaling, noise reduction, sharpening, and perspective correction.

## Installation

To set up the environment and run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/zhentaoliu409/opencv.git
   cd opencv
   ```

2. Install the required libraries:
   ```bash
   pip install opencv-python numpy matplotlib paddlepaddle paddleocr pytesseract
   ```

3. Ensure that [Tesseract](https://github.com/tesseract-ocr/tesseract) is installed and accessible via the command line.

## Usage

### Image Preprocessing
The image is preprocessed to improve text visibility by performing the following steps:
1. Rescaling and enhancing contrast and brightness.
2. Noise reduction and sharpening.
3. Morphological operations and edge detection to identify contours.

### Text Detection
Contours of possible text blocks are identified using the Canny edge detector, followed by filtering and sorting of these contours to extract valid text regions. Skewed text blocks are corrected using Hough transforms.

### Running OCR
You can run the system on a given image using the following script:

```bash
python ocr_lianxi.py --image/img_2.png --image/img_2.txt
```

### Performance Metrics
After text recognition, the system outputs the precision, recall, and F1 score, comparing the recognized text to a ground truth file.

## Performance Evaluation

The system measures the following metrics:
- **Precision**: The percentage of recognized characters that are correct.
- **Recall**: The percentage of characters in the image that were correctly recognized.
- **F1 Score**: A balance between precision and recall.

| OCR Engine  | Precision (%) | Recall (%) | F1 Score (%) | Time (s) |
|-------------|---------------|------------|--------------|----------|


## Results

### Example Outputs:
- **Tesseract Performance on `img_2.png`**:
  - Precision: 95.84%
  - Recall: 96.08%
  - F1 Score: 95.96%

- **PaddleOCR Performance on `img_2.png`**:
  - Precision: 90.22%
  - Recall: 90.44%
  - F1 Score: 90.33%

PaddleOCR tends to outperform Tesseract in more complex images, particularly those with skewed or noisy text, while Tesseract is faster in clean environments.

## References
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [Tesseract Documentation](https://github.com/tesseract-ocr/tesseract)
