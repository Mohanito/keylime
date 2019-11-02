'''
prepare_negative_dataset.py
    For every video in the input directory:
        Since we know this is a video without cats, we simply extract every frame.
    This is modified from prepare_dataset.py for convenience. Clearly there are better ways to implement this.
USAGE: python prepare_negative_dataset.py -i INPUT_DIR -o OUTPUT_DIR
'''

import numpy as np
import argparse
import imutils
import time
import cv2
import os


# Extract every frame from the video
def process_video(file_dir):
    # initialize the video stream and frame dimensions
    vs = cv2.VideoCapture(file_dir)
    (W, H) = (None, None)
    
    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
    except:
        print("[INFO] could not determine # of frames in video. No approx. completion time can be provided")
        total = -1

    np.random.seed(int(time.time()))        # Generate a seed for random output naming
    frame_counter = 0	                    # A counter for skipping frames
    
    while True:
	    # read the next frame
        (grabbed, frame) = vs.read()
	
	    # if not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

	    # if the frame dimensions are empty, grab them
        if W is None or H is None:
	        (H, W) = frame.shape[:2]

	    # print progress in this file for testing purpose
        frame_counter += 1
        print("Progress: {:.4f}%".format(frame_counter / total * 100))

        output_name = OUTPUT_PATH + "negative_" + str(np.random.randint(0, 999999)) + ".png"
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_name, grayscale)

    # release the file pointers
    print("[INFO] releasing cv2 VideoCapture for current file...")
    vs.release()



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input directory")
    ap.add_argument("-o", "--output", required=True, help="output directory")
    args = vars(ap.parse_args())

    INPUT_PATH = args["input"]
    OUTPUT_PATH = args["output"]

    total_files = len(os.listdir(INPUT_PATH))
    entries = os.scandir(INPUT_PATH)
    count = 1
    
    for entry in entries:
        process_video(entry.path)
        print("[INFO] Completed " + str(count) + " / " + str(total_files) + "files")
        count += 1
