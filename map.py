import numpy as np


def batch_iou(boxes1, boxes2):
    M, N = boxes1.shape[0], boxes2.shape[0]
    if N > M:
        boxes1, boxes2 = boxes2, boxes1
    iou_mat = np.zeros((M, N))
    for i, box in enumerate(boxes2):
        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        boxes1_area = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
        xx1 = np.maximum(box[0], boxes1[:, 0])
        yy1 = np.maximum(box[1], boxes1[:, 1])
        xx2 = np.minimum(box[2], boxes1[:, 2])
        yy2 = np.minimum(box[3], boxes1[:, 3])

        inter_w = np.maximum(xx2 - xx1 + 1, 0)
        inter_h = np.maximum(yy2 - yy1 + 1, 0)
        inter_area = inter_w * inter_h
        iou = inter_area / (area + boxes1_area - inter_area)
        iou_mat[:, i] = iou
    return iou_mat


def map_evaluation(pred_boxes, gt_boxes, iou_thresh=0.5):
    # iou
    scores = pred_boxes[:,4]
    sorted_idx = scores.argsort()[::-1]
    pred_boxes = pred_boxes[sorted_idx]
    iou_mat = batch_iou(pred_boxes[:,:4], gt_boxes)

    # precision and recall
    TP = 0
    FP = 0
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    precision = []
    recall = []
    for idx in range(len(pred_boxes)):
        ious = iou_mat[idx]
        gt_idx = np.argmax(ious)
        if ious[gt_idx] > iou_thresh:
            TP += 1
            iou_mat[:, gt_idx] = 0
        else:
            FP += 1
        precision.append(TP /(TP + FP))
        recall.append(TP/num_gt)
    print(precision)
    print(recall)

    # smooth precision
    smooth_precision = []
    reverse_max_p = 0
    for idx in range(len(pred_boxes)):
        temp_p = precision[num_pred - 1 - idx]
        reverse_max_p = max(reverse_max_p, temp_p)
        smooth_precision.insert(0, reverse_max_p)
    print(smooth_precision)
    print(recall)

    # average precision
    precision_at_recall = []
    precision_array = np.array(smooth_precision)
    recall_array = np.array(recall)
    for recall_level in np.linspace(0.0, 1.0, 11):
        x = precision_array[recall_array >= recall_level]
        if len(x) > 0:
            prec = max(x)
        else:
            prec = 0.0
        precision_at_recall.append(prec)
    avg_prec = np.mean(precision_at_recall)
    print('11 point precision is ', precision_at_recall)
    print('mAP is ', avg_prec)
    return avg_prec



if __name__ == '__main__':
    # [x1, y1, x2, y2, score]
    pred_boxes = np.array([
        [0, 0, 10, 10, 0.9],
        [0, 0, 9, 9, 0.8],
        [5, 5, 15, 15, 0.7],
        [20, 25, 30, 35, 0.6],
    ])
    gt_boxes = np.array([
        [0, 0, 10, 9],
        [5, 5, 15, 15],
    ])

    map = map_evaluation(pred_boxes, gt_boxes)
    print(map)