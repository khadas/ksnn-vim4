# Run

```sh
$ python3 yolov8n-picture.py --model ./models/VIM4/yolov8n_int8.adla --library ./libs/libnn_yolov8n.so --picture ./data/horses.jpg
$ python3 yolov8n-cap.py --model ./models/VIM4/yolov8n_int8.adla --library ./libs/libnn_yolov8n.so --device X
```

# Convert

# int8
```sh
$ ./convert \
--model-name yolov8n \
--model-type onnx \
--model ./yolov8n.onnx \
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
--model-name yolov8n \
--model-type onnx \
--model ./yolov8n.onnx \
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
--model-name yolov8n \
--model-type onnx \
--model ./yolov8n.onnx \
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
