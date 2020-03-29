# Pytorch model to ONNX model
python3 -u convert.py \
    --classify_model=ckpt/bird_cls_2720000.pth \
    --image_file=au1036.jpg \
    --output_onnx_model="classification.onnx"

# ONNX model to TF model
onnx-tf convert -i classification.onnx -o classification.pb

# TF model to TFLITE model
tflite_convert \
    --output_file=classification.tflite \
    --graph_def_file=classification.pb \
    --input_arrays=main_input \
    --output_arrays=main_output

# Test TFLITE model
python3 run_tflite.py \
    --image_file="au1036.jpg" \
    --tflite_model="classification.tflite"
