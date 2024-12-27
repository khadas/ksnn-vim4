import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

det_box_thresh = 0.2
min_size = 5
unclip_ratio = 1.5

character_str = ["blank"]
with open("./data/ppocr_keys_v1.txt", "rb") as fin:
    lines = fin.readlines()
    for line in lines:
        line = line.decode("utf-8").strip("\n").strip("\r\n")
        character_str.append(line)
character_str.append(" ")
ignored_token = [0]


def ocr_det_postprocess(det_output, original_image, det_input_size):
	outs = cv2.findContours((det_output * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	if len(outs) == 3:
		contours = outs[1]
	elif len(outs) == 2:
		contours = outs[0]
	
	det_results = []
	for i in range(len(contours)):
		bounding_box = cv2.boundingRect(contours[i])
		if bounding_box[2] < min_size or bounding_box[3] < min_size:
			continue
		
		mask = np.ones((bounding_box[3], bounding_box[2]), dtype=np.uint8)
		tmp_det_output = det_output.reshape(det_input_size[0], det_input_size[1])
		score = cv2.mean(tmp_det_output[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]], mask)[0]
		if score < det_box_thresh:
			continue
		
		box = np.array(((bounding_box[0], bounding_box[1]),
                        (bounding_box[0] + bounding_box[2], bounding_box[1]),
                        (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                        (bounding_box[0], bounding_box[1] + bounding_box[3])))
        
		poly = Polygon(box)
		distance = poly.area * unclip_ratio / poly.length
		offset = pyclipper.PyclipperOffset()
		offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
		expanded = offset.Execute(distance)
		tmp_box = np.array(expanded)
        
		xmin = max(int(np.min(tmp_box[0, :, 0]) / det_input_size[1] * original_image.shape[1]), 0)
		ymin = max(int(np.min(tmp_box[0, :, 1]) / det_input_size[0] * original_image.shape[0]), 0)
		xmax = min(int(np.max(tmp_box[0, :, 0]) / det_input_size[1] * original_image.shape[1] + 1), original_image.shape[1])
		ymax = min(int(np.max(tmp_box[0, :, 1]) / det_input_size[0] * original_image.shape[0] + 1), original_image.shape[0])
        
		det_results.append([xmin, ymin, xmax, ymax, score, 0])
        
	return det_results

def ocr_rec_postprocess(rec_output):
    rec_idx = rec_output.argmax(axis=1)
    rec_prob = rec_output.max(axis=1)
    
    selection = np.ones(len(rec_idx), dtype=bool)
    selection[1:] = rec_idx[1:] != rec_idx[:-1]
    selection &= rec_idx != ignored_token
    
    char_list = [character_str[text_id] for text_id in rec_idx[selection]]
    character_result = "".join(char_list)
    
    return character_result


