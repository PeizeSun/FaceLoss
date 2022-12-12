import numpy as np


def NMS(boxes, iou_thresh=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_id = scores.argsort()[::-1]

    keep_ind = []
    while len(sorted_id) > 0:
        temp_id = sorted_id[0]
        keep_ind.append(temp_id)

        xx1 = np.maximum(x1[temp_id], x1[sorted_id[1:]])
        yy1 = np.maximum(y1[temp_id], y1[sorted_id[1:]])
        xx2 = np.minimum(x2[temp_id], x2[sorted_id[1:]])
        yy2 = np.minimum(y2[temp_id], y2[sorted_id[1:]])

        inter_w = np.maximum(xx2 - xx1 + 1, 0)
        inter_h = np.maximum(yy2 - yy1 + 1, 0)
        inter_areas = inter_w * inter_h
        iou = inter_areas / (areas[temp_id] + areas[sorted_id[1:]] - inter_areas)

        temp_keep = iou < iou_thresh
        sorted_id = sorted_id[1:][temp_keep]

    return keep_ind


if __name__ == '__main__':
    # [x1, y1, x2, y2, score]
    boxes = np.array([
        [0, 0, 10, 10, 0.9],
        [0, 0, 9, 9, 0.8],
        [5, 5, 15, 15, 0.7],
        [20, 25, 30, 35, 0.6],
    ])
    keep = NMS(boxes)
    res_boxes = boxes[keep]
    print(res_boxes)
