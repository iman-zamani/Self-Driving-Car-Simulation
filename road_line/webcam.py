import cv2
import numpy as np
import os
from tensorflow.lite.python.interpreter import Interpreter

def live_tflite_detection(model_directory, min_conf=0.9):
    model_path = os.path.join(model_directory, 'detect.tflite')
    label_path = os.path.join(model_directory, 'labelmap.txt')

    # Check if model and label files exist
    if not os.path.exists(model_path) or not os.path.exists(label_path):
        print("Model or label file does not exist.")
        return

    # Load the label map
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TensorFlow Lite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Resize frame to expected shape [1xHxWx3]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            if float_input:
                input_data = (np.float32(input_data) - input_mean) / input_std

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            boxes = interpreter.get_tensor(output_details[1]['index'])[0]
            classes = interpreter.get_tensor(output_details[3]['index'])[0]
            scores = interpreter.get_tensor(output_details[0]['index'])[0]

            for i in range(len(scores)):
                if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                    ymin, xmin, ymax, xmax = [int(max(1, box * dim)) for box, dim in zip(boxes[i], [imH, imW, imH, imW])]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                    object_name = labels[int(classes[i])]
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# User selects the model
model_directory = '.'
live_tflite_detection(model_directory)
