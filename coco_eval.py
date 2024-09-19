import json
import os
import argparse
import torch
import yaml
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate  # For printing the confusion matrix

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def compute_iou_matrix(boxes1, boxes2):
    # boxes1: (N1, 4)
    # boxes2: (N2, 4)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N1,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (N2,)

    x1 = np.maximum(boxes1[:, np.newaxis, 0], boxes2[np.newaxis, :, 0])  # (N1, N2)
    y1 = np.maximum(boxes1[:, np.newaxis, 1], boxes2[np.newaxis, :, 1])
    x2 = np.minimum(boxes1[:, np.newaxis, 2], boxes2[np.newaxis, :, 2])
    y2 = np.minimum(boxes1[:, np.newaxis, 3], boxes2[np.newaxis, :, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h  # (N1, N2)

    union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area  # (N1, N2)
    iou_matrix = inter_area / (union_area + 1e-6)
    return iou_matrix


def compute_confusion_matrix(coco_gt, pred_json_path, num_classes, iou_threshold=0.5):
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    # Load predictions
    with open(pred_json_path, 'r') as f:
        predictions = json.load(f)

    # Create a dictionary mapping image_id to list of predictions
    pred_by_image = {}
    for pred in predictions:
        image_id = pred['image_id']
        pred_by_image.setdefault(image_id, []).append(pred)

    # Get list of image_ids
    image_ids = coco_gt.getImgIds()

    for image_id in tqdm(image_ids):
        # Get ground truth annotations for this image
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x1, y1, w, h]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # Convert to [x1, y1, x2, y2]
            gt_boxes.append(bbox)
            gt_labels.append(ann['category_id'] - 1)  # category_id starts from 1, labels from 0

        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        num_gts = len(gt_boxes)
        gt_matched = np.zeros(num_gts, dtype=bool)

        # Get predictions for this image
        preds = pred_by_image.get(image_id, [])
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        for pred in preds:
            bbox = pred['bbox']  # [x1, y1, w, h]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # Convert to [x1, y1, x2, y2]
            pred_boxes.append(bbox)
            pred_labels.append(pred['category_id'] - 1)  # category_id starts from 1, labels from 0
            pred_scores.append(pred['score'])

        pred_boxes = np.array(pred_boxes)
        pred_labels = np.array(pred_labels)
        num_preds = len(pred_boxes)
        pred_matched = np.zeros(num_preds, dtype=bool)

        # Match predictions to ground truths
        if num_gts > 0 and num_preds > 0:
            # Compute IoU between all predictions and ground truths
            iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
            for i in range(num_preds):
                # For each prediction, find the ground truth with highest IoU
                ious = iou_matrix[i]
                max_iou = np.max(ious)
                max_j = np.argmax(ious)
                if max_iou >= iou_threshold and not gt_matched[max_j]:
                    gt_matched[max_j] = True
                    pred_matched[i] = True
                    gt_label = gt_labels[max_j]
                    pred_label = pred_labels[i]
                    confusion_matrix[gt_label, pred_label] += 1
                else:
                    # False positive
                    pred_label = pred_labels[i]
                    confusion_matrix[num_classes, pred_label] += 1
        elif num_preds > 0:
            # No ground truths, all detections are false positives
            for i in range(num_preds):
                pred_label = pred_labels[i]
                confusion_matrix[num_classes, pred_label] += 1
        elif num_gts > 0:
            # No detections, all ground truths are missed
            for j in range(num_gts):
                gt_label = gt_labels[j]
                confusion_matrix[gt_label, num_classes] += 1

        # For unmatched ground truths (missed detections)
        for j in range(num_gts):
            if not gt_matched[j]:
                gt_label = gt_labels[j]
                confusion_matrix[gt_label, num_classes] += 1

        # For unmatched detections (false positives)
        for i in range(num_preds):
            if not pred_matched[i]:
                pred_label = pred_labels[i]
                confusion_matrix[num_classes, pred_label] += 1

    return confusion_matrix


def print_confusion_matrix(confusion_matrix, class_names):
    """
    confusion_matrix: numpy array of shape (num_classes+1, num_classes+1)
    class_names: list of class names, length num_classes
    """
    num_classes = len(class_names)
    labels = class_names + ['background']
    cm = confusion_matrix.astype('float')
    # Normalize the confusion matrix by row (ground truth labels)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm, row_sums, where=row_sums != 0)
    # Replace NaNs with zeros if any row sums are zero
    cm_normalized = np.nan_to_num(cm_normalized)
    # Format the confusion matrix as a list of lists
    table = []
    for i in range(num_classes + 1):
        row = [f"{cm_normalized[i, j]:.2f}" for j in range(num_classes + 1)]
        table.append(row)
    # Prepare headers
    headers = labels
    # Print the table
    print("Normalized Confusion Matrix:")
    print("Rows: Ground Truth")
    print("Columns: Predicted")
    print(tabulate(table, headers=headers, showindex=labels, tablefmt="grid"))


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef],
                                                         mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    SET_NAME = params['val_set']
    VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')

    # Compute and print confusion matrix
    confusion_matrix = compute_confusion_matrix(coco_gt, f'{SET_NAME}_bbox_results.json', len(obj_list))
    print_confusion_matrix(confusion_matrix, obj_list)
