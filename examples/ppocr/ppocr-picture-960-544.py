import numpy as np
import os
import urllib.request
import argparse
import sys
import math
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time
from postprocess import ocr_det_postprocess, ocr_rec_postprocess
from PIL import Image, ImageDraw, ImageFont

det_mean = [123.675, 116.28, 103.53]
det_var = [255 * 0.229, 255 * 0.224, 255 * 0.225]
rec_mean = 127.5
rec_var = 128

det_input_size = (544, 960) # (model height, model width)
rec_input_size = ( 48, 320) # (model height, model width)
rec_output_size = (40, 97)

font = ImageFont.truetype("./data/simfang.ttf", 20)

def draw(image, boxes):
    draw_img = Image.fromarray(image)
    draw = ImageDraw.Draw(draw_img)
    for box in boxes:
        x1, y1, x2, y2, score, text = box
        left = max(0, np.floor(x1 + 0.5).astype(int))
        top = max(0, np.floor(y1 + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x2 + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y2 + 0.5).astype(int))

        draw.rectangle((left, top, right, bottom), outline=(0, 255, 0), width=2)
        draw.text((left, top - 20), text, font=font, fill=(0, 255, 0))
    
    return draw_img, np.array(draw_img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--det_library", help="Path to C static library file for ppocr_det")
    parser.add_argument("--det_model", help="Path to nbg file for ppocr_det")
    parser.add_argument("--rec_library", help="Path to C static library file for ppocr_rec")
    parser.add_argument("--rec_model", help="Path to nbg file for ppocr_rec")
    parser.add_argument("--picture", help="Path to input picture")
    parser.add_argument("--level", help="Information printer level: 0/1/2")

    args = parser.parse_args()
    if args.det_model :
        if os.path.exists(args.det_model) == False:
            sys.exit('ppocr_det Model \'{}\' not exist'.format(args.det_model))
        det_model = args.det_model
    else :
        sys.exit("NBG file not found !!! Please use format: --det_model")
    if args.rec_model :
        if os.path.exists(args.rec_model) == False:
            sys.exit('ppocr_det Model \'{}\' not exist'.format(args.rec_model))
        rec_model = args.rec_model
    else :
        sys.exit("NBG file not found !!! Please use format: --rec_model")
    
    if args.picture :
        if os.path.exists(args.picture) == False:
            sys.exit('Input picture \'{}\' not exist'.format(args.picture))
        picture = args.picture
    else :
        sys.exit("Input picture not found !!! Please use format: --picture")
    
    if args.det_library :
        if os.path.exists(args.det_library) == False:
            sys.exit('ppocr_det C static library \'{}\' not exist'.format(args.det_library))
        det_library = args.det_library
    else :
        sys.exit("ppocr_det C static library not found !!! Please use format: --det_library")
    if args.rec_library :
        if os.path.exists(args.rec_library) == False:
            sys.exit('ppocr_rec C static library \'{}\' not exist'.format(args.rec_library))
        rec_library = args.rec_library
    else :
        sys.exit("ppocr_rec C static library not found !!! Please use format: --rec_library")
    
    if args.level == '1' or args.level == '2' :
        level = int(args.level)
    else :
        level = 0

    ppocr_det = KSNN('VIM4')
    ppocr_rec = KSNN('VIM4')
    print(' |---+ KSNN Version: {} +---| '.format(ppocr_det.get_nn_version()))

    print('Start init neural network ...')
    ppocr_det.nn_init(library=det_library, model=det_model, level=level)
    ppocr_rec.nn_init(library=rec_library, model=rec_model, level=level)
    print('Done.')

    print('Get input data ...')

    orig_img = cv.imread(picture, cv.IMREAD_COLOR)
    det_img = cv.resize(orig_img, (det_input_size[1], det_input_size[0])).astype(np.float32)
    det_img[:, :, 0] = (det_img[:, :, 0] - det_mean[0]) / det_var[0]
    det_img[:, :, 1] = (det_img[:, :, 1] - det_mean[1]) / det_var[1]
    det_img[:, :, 2] = (det_img[:, :, 2] - det_mean[2]) / det_var[2]
    
    print('Done.')

    print('Start inference ...')
    start = time.time()
    det_output = ppocr_det.nn_inference(det_img, input_shape=(det_input_size[0], det_input_size[1], 3), input_type="RAW", output_shape=[(det_input_size[0], det_input_size[1], 1)], output_type="FLOAT")
    
    det_results = ocr_det_postprocess(det_output[0], orig_img, det_input_size)
    
    final_results = []
    
    for i in range(len(det_results)):
        xmin, ymin, xmax, ymax, _, _ = det_results[i]
        rec_img = orig_img[ymin:ymax, xmin:xmax]
        
        new_height = rec_input_size[0]
        new_width = int(new_height / rec_img.shape[0] * rec_img.shape[1])
        
        if new_width > rec_input_size[1] * 1.2:
            # text too long. If you want to detect it, please convert rec model input longer.
            continue
        elif new_width < rec_input_size[1] * 1.2 and new_width > rec_input_size[1]:
            new_width = rec_input_size[1]        
        
        rec_img = cv.resize(rec_img, (new_width, new_height)).astype(np.float32)
        rec_img = (rec_img - rec_mean) / rec_var
        padding_img = np.zeros((rec_input_size[0], rec_input_size[1], 3)).astype(np.float32)
        padding_img[:, :new_width] = rec_img
        
        rec_output = ppocr_rec.nn_inference(padding_img, input_shape=(rec_input_size[0], rec_input_size[1], 3), input_type="RAW", output_shape=[(rec_output_size[0], rec_output_size[1])], output_type="FLOAT")
        
        det_results[i][5] = ocr_rec_postprocess(rec_output[0])
        final_results.append(det_results[i])
    
    end = time.time()
    print('Done. inference time: ', end - start)

    if det_results is not None:
        pil_img, cv_img = draw(orig_img, final_results)

    ppocr_det.nn_destory_network()
    ppocr_rec.nn_destory_network()
    cv.imwrite("./result.jpg", cv_img)
    cv.imshow("results", cv_img)
    cv.waitKey(0)
