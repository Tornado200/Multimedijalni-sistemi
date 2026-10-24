import sys
import cv2
import numpy as np
from common import get_src_dst, OUT
from step2_preprocessing import process_image

def perspective_transform(image_path):
    binary_img = process_image(image_path)
    if binary_img is None:
        print(f"Error: Could not load image from {image_path}.")
        return None

    height, width = binary_img.shape[:2]

    src, dst = get_src_dst(width, height)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(binary_img, M, (width, height), flags=cv2.INTER_LINEAR)

    cv2.imwrite(f"{OUT}/output_warped.jpg", warped)
    print("Perspective transform completed.")
    return warped, M, Minv

if __name__ == "__main__":
    perspective_transform(sys.argv[1] if len(sys.argv) > 1 else 'test_images/test2.jpg')