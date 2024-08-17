import numpy as np
import tensorflow as tf
import cv2

# Define input_height and input_width based on the model's input size
input_height = 224
input_width = 224

# Load TFLite model and perform inference
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the webcam (0 represents the default camera, change it to the appropriate index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize and normalize the frame to the desired input size of the model
    image = tf.image.resize(frame_rgb, [input_height, input_width])
    image = image / 255.0  # Normalize the image to [0, 1], this keeps it as FLOAT32
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Set the input tensor data (should be FLOAT32 now)
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor and post-process the results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)

    # Display the predicted class on the frame
    name = 'no Sign';
    if predicted_class == 0:
        name = 'Traffic Sign: Do Not Enter'
    elif predicted_class == 1:
        name = 'Traffic Sign: Directional Arrow'
    elif predicted_class == 2:
        name = 'no Sign'
    elif predicted_class == 3:
        name = 'Traffic Sign: Stop '
    elif predicted_class == 4:
        name = 'Traffic Sign: Dead End'
    
    cv2.putText(frame, f"{name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

