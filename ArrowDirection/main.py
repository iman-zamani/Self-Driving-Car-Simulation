# Import packages
import cv2
import numpy as np
import time
from tensorflow.lite.python.interpreter import Interpreter
import math
### Define function for inferencing with TFLite model and displaying results

def tflite_detect_webcam(modelpath, lblpath, min_conf=0.9):

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Open webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    # Loop over frames from the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        sar_x = 0
        sar_y = 0
        tah_x = 0
        tah_y = 0
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                if (int(classes[i])) == 0 :
                    sar_x = (xmin + xmax) / 2
                    sar_y = (ymax + ymin) / 2
                else :
                    tah_x = (xmin + xmax) / 2
                    tah_y = (ymax + ymin) / 2
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if abs(sar_x - tah_x) >= abs(sar_y - tah_y):
            if sar_x >= tah_x :
                print("right")
            else :
                print("left")
        else :
            if sar_y >= tah_y:
                print("down")
            else :
                print("up")
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 10:  # Calculate FPS every 10 frames
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            start_time = end_time
            frame_count = 0

        # Display the frame with detections
        cv2.imshow("Object Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return

# Call the function to perform object detection using the webcam
model_path = 'detect.tflite'
label_path = 'labelmap.txt'
tflite_detect_webcam(model_path, label_path)
