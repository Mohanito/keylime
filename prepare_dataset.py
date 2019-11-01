'''
prepare_dataset.py
    For every video in the input directory:
        examine 1 out of every 20 frames
        detect frames with YOLOv3 pretrained model with a high confidence
        convert the frame to grayscale, and save it to output directory
USAGE: python prepare_dataset.py -i INPUT_DIR -o OUTPUT_DIR
'''

import numpy as np
import argparse
import imutils
import time
import cv2
import os


# --------------------------------- GLOBAL VARS ---------------------------------------- #
CONFIDENCE = 0.8                    # Controls the quality of filtered frames
DOWN_SAMPLING = 20                  # 1 out of DOWN_SAMPLING frames will be processed
CLASS_CAT = 15                      # See yolo-coco/coco.names



def process_video(file_dir):
    # initialize the video stream and frame dimensions
    vs = cv2.VideoCapture(file_dir) # FIXME
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

	    # Downsampling check
        frame_counter += 1
        if frame_counter % DOWN_SAMPLING != 0:
            continue
        print("Progress: {:.4f}%".format(frame_counter / total * 100))

	    # construct a blob from the input frame and then perform a forward
	    # pass of the YOLO object detector, giving us our bounding boxes
	    # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		    swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # Evaluate results
        for output in layerOutputs:
            for detection in output:
                # extract the class ID and confidence of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions, and make sure it is a CAT prediction
                if confidence > CONFIDENCE and classID == CLASS_CAT:
                    print("[CAT DETECTED] Saving the current frame...")
                    output_name = OUTPUT_PATH + "cat_" + str(np.random.randint(0, 999999)) + ".png"
                    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(output_name, grayscale)

        # estimating total time (only once at the start of each file)
        if total > 0 and frame_counter == DOWN_SAMPLING:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total / DOWN_SAMPLING))

    # release the file pointers
    print("[INFO] cleaning up...")
    vs.release()



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input directory")
    ap.add_argument("-o", "--output", required=True, help="output directory")
    args = vars(ap.parse_args())

    INPUT_PATH = args["input"]
    OUTPUT_PATH = args["output"]

    # Setup YOLO
    labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
    configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] Loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    total_files = len(os.listdir(INPUT_PATH))
    entries = os.scandir(INPUT_PATH)
    count = 1
    
    for entry in entries:
        process_video(entry.path)
        print("[INFO] Completed " + str(count) + " / " + str(total_files) + "files")
        count += 1
