import os
import cv2
import time
import traceback
import threading
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, use_gpu=True)
stop_event = threading.Event()

def captcha_ocr():
    folder_path = "Inp_images/"
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    while not stop_event.is_set():
        try:
            image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]
            
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Skipping {image_file}, unable to read image.")
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply denoising
                denoised_Stage1 = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=4, searchWindowSize=21)
                denoised_Stage2 = cv2.fastNlMeansDenoising(denoised_Stage1, None, 31, 12, 21)
                denoised_Stage3 = cv2.fastNlMeansDenoising(denoised_Stage2, h=10, templateWindowSize=6, searchWindowSize=21)

                # Apply binary thresholding
                _, binary = cv2.threshold(denoised_Stage3, 128, 255, cv2.THRESH_BINARY_INV)

                # Morphological processing
                kernel = np.ones((3, 3), np.uint8)
                cleaned_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # OCR Processing
                results = ocr.ocr(cleaned_image, det=False)
                
                extracted_text = ''.join(filter(str.isalnum, (results[0][0])[0])) if results and results[0] else "unknown"
                print(f"Extracted CAPTCHA: {extracted_text}")
                output_folder = "Captcha"
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f"{extracted_text}-{image_file}")
                cv2.imwrite(output_path, image)
                print(f"Captcha saved as {output_path}")

                # Delete the processed image
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Deleted {image_file}")
                else:
                    print("File not found for deletion.")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Stopping thread...")
            stop_event.set()

        # Wait before checking the folder again
        time.sleep(2)

# Start the thread
thread = threading.Thread(target=captcha_ocr, daemon=True)
thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping OCR processing...")
    traceback.print_exc() 
    stop_event.set()
    thread.join()
    print("Thread stopped successfully.")
