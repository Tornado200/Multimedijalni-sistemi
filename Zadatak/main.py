import argparse

from common import OUT
from step3_perspective import perspective_transform
from step4_lane_fit import find_lane_pixels_and_fit
from step5_curvature_and_draw import measure_curvature_and_draw
from video_pipeline import process_video


def main():
    parser = argparse.ArgumentParser(description="Advanced Lane Finding Pipeline")
    parser.add_argument("mode", choices=["image", "video"], help="image or video mode")
    parser.add_argument("--input", default=None, help="path to the input image or video")
    parser.add_argument("--max-frames", type=int, default=None, help="video mode only: process just the first frames")
    args = parser.parse_args()

    if args.mode == "image":
        path = args.input or "test_images/test1.jpg"
        if perspective_transform(path) is None:
            return
        find_lane_pixels_and_fit(f"{OUT}/output_warped.jpg")
        measure_curvature_and_draw(path, f"{OUT}/output_warped.jpg")
    else:
        path = args.input or "test_videos/project_video03.mp4"
        process_video(path, args.max_frames)


if __name__ == "__main__":
    main()