import sys
import cv2
import numpy as np
from common import OUT, get_src_dst


def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    white = l_channel >= 190
    yellow = (h_channel >= 15) & (h_channel <= 35) & (s_channel >= 80) & (l_channel >= 60)

    combined_binary = np.zeros(l_channel.shape, dtype=np.uint8)
    combined_binary[white | yellow] = 255

    h, w = img.shape[:2]
    src, _ = get_src_dst(w, h)
    mask = np.zeros_like(combined_binary)
    cv2.fillPoly(mask, [src.astype(np.int32)], 255)
    masked_binary = cv2.bitwise_and(combined_binary, mask)

    output_path = f'{OUT}/output_binary.jpg'
    cv2.imwrite(output_path, masked_binary)
    print(f"Preprocessing completed on {image_path}. Saved to '{output_path}'.")
    return masked_binary


if __name__ == "__main__":
    process_image(sys.argv[1] if len(sys.argv) > 1 else 'test_images/test2.jpg')