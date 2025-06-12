# Run

```sh
$ python3 yolov8n_pose-picture.py --model ./models/VIM4/yolov8n_pose_int8.adla --library ./libs/libnn_yolov8n_pose.so --picture ./data/horses.jpg
$ python3 yolov8n_pose-cap.py --model ./models/VIM4/yolov8n_pose_int8.adla --library ./libs/libnn_yolov8n_pose.so --type usb --device 0
$ python3 yolov8n_pose-cap.py --model ./models/VIM4/yolov8n_pose_int8.adla --library ./libs/libnn_yolov8n_pose.so --type mipi --device 63
```

# Convert

# int8
```sh
$ ./convert \
--model-name yolov8n_pose \
--model-type onnx \
--model ./yolov8n_pose.onnx \
--inputs "images" \
--input-shapes "3,640,640" \
--dtype "float32" \
--quantize-dtype int8 \
--outdir onnx_output \
--inference-input-type "float32" \
--inference-output-type "float32" \
--channel-mean-value '0,0,0,255' \
--source-file dataset.txt \
--iterations 1 \
--batch-size 1 \
--kboard VIM4
```

# uint8
```sh
$ ./convert \
--model-name yolov8n_pose \
--model-type onnx \
--model ./yolov8n_pose.onnx \
--inputs "images" \
--input-shapes "3,640,640" \
--dtype "float32" \
--quantize-dtype uint8 \
--outdir onnx_output \
--inference-input-type "float32" \
--inference-output-type "float32" \
--channel-mean-value '0,0,0,255' \
--source-file dataset.txt \
--iterations 1 \
--batch-size 1 \
--kboard VIM4
```

# int16
```sh
$ ./convert \
--model-name yolov8n_pose \
--model-type onnx \
--model ./yolov8n_pose.onnx \
--inputs "images" \
--input-shapes "3,640,640" \
--dtype "float32" \
--quantize-dtype int16 \
--outdir onnx_output \
--inference-input-type "float32" \
--inference-output-type "float32" \
--channel-mean-value '0,0,0,255' \
--source-file dataset.txt \
--iterations 1 \
--batch-size 1 \
--kboard VIM4
```
