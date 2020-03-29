import cv2
import argparse
import numpy as np
import tensorflow as tf


def main(args):
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    print("TFLITE input shape:", input_shape)
    img = cv2.imread(args.image_file)
    img = cv2.resize(img, (300, 300))
    img = np.transpose(img, (2, 0, 1))
    img = img.reshape((1, 3, 300, 300)).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], img)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    print("TFLITE output", output_data, output_data.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', default=None, type=str, help='Image file to be predicted')
    parser.add_argument('--tflite_model', default=None, type=str, help='TFLITE model file')
    args = parser.parse_args()

    main(args)
