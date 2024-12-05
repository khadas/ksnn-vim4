# Run

```sh
$ python3 ppocr-picture.py --det_model ./models/VIM4/ppocr_det_int8.adla --det_library ./libs/libnn_ppocr_det.so --rec_model ./models/VIM4/ppocr_rec_int16.adla --rec_library ./libs/libnn_ppocr_rec.so --picture ./data/test.png
$ python3 ppocr-cap.py --det_model ./models/VIM4/ppocr_det_int8.adla --det_library ./libs/libnn_ppocr_det.so --rec_model ./models/VIM4/ppocr_rec_int16.adla --rec_library ./libs/libnn_ppocr_rec.so --device X
```

# Convert

# ppocr_det
```sh
$ ./convert \
--model-name ppocr_det \
--model-type onnx \
--model ./ppocr_det.onnx \
--inputs "x" \
--input-shapes "3,736,736" \
--dtype "float32" \
--quantize-dtype int8 \
--outdir onnx_output
--channel-mean-value '123.675,116.28,103.53,57.375' \
--inference-input-type "float32" \
--inference-output-type "float32" \
--source-file dataset.txt \
--iterations 1 \
--batch-size 1 \
--kboard VIM4
```


# ppocr_rec
```sh
$ ./convert \
--model-name ppocr_rec \
--model-type onnx \
--model ./ppocr_rec.onnx \
--inputs "x" \
--input-shapes "3,48,320" \
--dtype "float32" \
--quantize-dtype int16 \
--outdir onnx_output
--channel-mean-value '127.5,127.5,127.5,128' \
--inference-input-type "float32" \
--inference-output-type "float32" \
--source-file dataset.txt \
--iterations 1 \
--batch-size 1 \
--kboard VIM4
```


