from matplotlib.figure import NonGuiException
from PIL import Image
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import numpy as np
from brisque import BRISQUE
import sewar
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import pandas as pd
import seaborn as sns
from scipy import interpolate
import matplotlib.pyplot as plt

class ConGenVismoEval():
  def __init__(self,
              dtr_resnet = "101"):
    """Loads the respective models required for the evaluation.
    For element presense please use dtr_resnet = '50', and for
    iou please use '101'.
    """
    self.__clip = SentenceTransformer('clip-ViT-B-32')
    self.__brisque = BRISQUE(url=False)
    self.__dtr_processor = processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-{dtr_version}")
    self.__dtr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-{dtr_version}")
  
  def read_img(self,
               img):
    return np.array(Image.open(img).convert("RGB"))

  def resize(self,
             img_np,
             size):
    return sewar.no_ref.imresize(img_np, (size[1],size[0]))
  
  def read_same_size(self,
                     img_yt,
                     img_yhat):
    img_yt_np = self.read_img(img_yt)
    imp_yhat_np = self.resize(self.read_img(img_yhat), img_yt_np.shape[:2])
    return img_yt_np, imp_yhat_np
  
  def clip_sim(self,
               img_yt,
               img_yhat):
    """Gets two respective image paths, predicted and ground-truth and
    returns the respective clip similarity score.
    """
    embedding_yt = self.__clip.encode(Image.open(img_yt))
    embedding_yhat = self.__clip.encode(Image.open(img_yhat))
    clip_cos_score = util.cos_sim(embedding_yt, embedding_yhat)
    return clip_cos_score.detach().numpy()[0][0]
  
  def psnr(self,
           img_yt,
           img_yhat):
    img_yt_np, imp_yhat_np = self.read_same_size(img_yt, img_yhat)
    return sewar.psnr(img_yt_np,imp_yhat_np)
  
  def brisque(self,
              img_yhat):
    return self.__brisque.score(Image.open(img_yhat))
  
  def ssim(self,
           img_yt,
           img_yhat):
    img_yt_np, imp_yhat_np = self.read_same_size(img_yt, img_yhat)
    ssim, cs = sewar.ssim(img_yt_np, imp_yhat_np)
    return {"ssim": ssim,
            "cs": cs}

  def uqi(self,
          img_yt,
          img_yhat):
    img_yt_np, imp_yhat_np = self.read_same_size(img_yt, img_yhat)
    return sewar.full_ref.uqi(img_yt_np, imp_yhat_np)

  def IoU(self,
          bbox_a,
          bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    intersection_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
    if intersection_area == 0:
        return 0

    box_area_a = abs((bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    box_area_b = abs((bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))

    iou = intersection_area / float(box_area_a + box_area_b - intersection_area)
    
    return iou.detach().numpy()
  
  def object_detector(self,
                      img
                      ):
    img = Image.open(img).convert("RGB")
    inputs = self.__dtr_processor(images=img, return_tensors="pt")
    outputs = self.__dtr(**inputs)

    target_sizes = torch.tensor([img.size[::-1]])
    results = self.__dtr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    elements = [self.__dtr.config.id2label[i.item()] for i in results["labels"]]
    elements_counted = Counter(elements)
    return elements_counted, results
  
  def element_presence_pr(self,
                          elements_yt,
                          elements_yhat):
    if sum(elements_yhat.values()) == 0:
      if len(elements_yt & elements_yhat) == 0:
        return 1
      else:
        return 0
    return len(elements_yt & elements_yhat) / sum(elements_yhat.values())
  
  def element_presence_re(self,
                          elements_yt,
                          elements_yhat):
    if sum(elements_yt.values()) == 0:
      if len(elements_yt & elements_yhat) == 0:
        return 1
      else:
        return 0
    return len(elements_yt & elements_yhat) / sum(elements_yt.values())
  
  def element_presence_f1(self,
                          element_presence_pr,
                          element_presence_re):
    if (element_presence_pr + element_presence_re) == 0:
      return 0
    return ((element_presence_pr * element_presence_re)*2)/(element_presence_pr + element_presence_re)
  
  def interpolation(self,
                    y):
    if len(y) > 2:
      kind = "linear"
    elif len(y) == 1:
      return [], [], np.array([y for _ in range(20)]).reshape((-1))
    else:
      kind = "linear"
    x = np.linspace(0, 1,len(y))
    x_new = np.linspace(0, 1, 20)
    f2 = interpolate.interp1d(x, y, kind = kind)
    return x, x_new, f2(x_new)
  
  def draw(self,
           score_name):
    all = []
    for i in all_scores.keys():
      if type(score_name) != list:
        y = all_scores[i][score_name]
        y_axis = score_name
      else:
        y = [_[score_name[1]] for _ in all_scores[i][score_name[0]]]
        y_axis = score_name[1]
      if len(y) > 0:
        x, x_new, y_new = self.interpolation(y)
        all.append(y_new)
    all_np = np.array(all)
    df = pd.DataFrame(all_np).melt()
    df.columns = ["hop", y_axis]
    fig, ax = plt.subplots()
    sns.lineplot(x="hop", y=y_axis, data=df, ax = ax)
    ax.set_xlim(0,19)
    ax.set_xticks(range(1,20,2))
    ax.set_xticklabels([round(i*0.1,1) for i in range(1,11)])
    plt.savefig(y_axis+".pdf")
    plt.show()
  
  def element_presence_scores(self,
                              img_yt,
                              img_yhat):
    elements_yt , bbox_yt = self.object_detector(img_yt)
    elements_yhat , bbox_yhat = self.object_detector(img_yhat)
    return {
        "precision": (pr := self.element_presence_pr(elements_yt, elements_yhat)),
        "recall": (re := self.element_presence_re(elements_yt, elements_yhat)),
        "f1": self.element_presence_f1(pr,re),
        "elements": {
            "img_yt": elements_yt,
            "img_yhat": elements_yhat
        }
    }
  def get_bbox(self,img):
    c, result = self.object_detector(img)
    size = Image.open(img).convert("RGB").size
    grouped_result = {
    }
    for label, box in zip(result["labels"].detach().numpy(),
                          result["boxes"].detach().numpy()):
      r_label = eval._ConGenVismoEval__dtr.config.id2label[label]
      if label in grouped_result.keys():
        grouped_result[r_label].append(box)
      else:
        grouped_result[r_label] = [box]
    new_grouped_result = {}
    for key in grouped_result.keys():
      boxes = []
      for box in grouped_result[key]:
        box = np.array([box[0]/size[0],
                        box[1]/size[1],
                        box[2]/size[0],
                        box[3]/size[1]])
        boxes.append((box*512).astype(int))
      new_grouped_result[key] = boxes
    return new_grouped_result

def iou_score(self,
            img_yt,
            img_yh):
  results_yt = get_bbox(img_yt)
  resutls_yh = get_bbox(img_yh)
  common_objects = set(results_yt.keys()).intersection(set(resutls_yh.keys()))

  iou_common = []
  if len(common_objects) == 0:
    iou_common = [0]
  else:
    for common_object in common_objects:
      zero_yt = np.zeros((512,512))
      zero_yh = np.zeros((512,512))
      for bbox in results_yt[common_object]:
        zero_yt[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
      for bbox in resutls_yh[common_object]:
        zero_yh[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
      c = np.sum((zero_yh * zero_yt)> 0)
      s = np.sum((zero_yh + zero_yt)> 0)
      iou_common.append(c/s)

  iou_common = np.mean(iou_common) if len(iou_common) > 0 else 0.0

  iou_precision = []
  common_objects = set(resutls_yh.keys())
  objects_yt = set(results_yt.keys())
  for common_object in common_objects:
    if common_object in objects_yt:
      zero_yt = np.zeros((512,512))
      zero_yh = np.zeros((512,512))
      for bbox in results_yt[common_object]:
        zero_yt[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
      for bbox in resutls_yh[common_object]:
        zero_yh[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
      c = np.sum((zero_yh * zero_yt)> 0)
      s = np.sum((zero_yh + zero_yt)> 0)
      iou_precision.append(c/s)
    else:
      iou_precision.append(0)
  iou_precision = np.mean(iou_precision) if len(iou_precision) > 0 else 0.0
  
  iou_recall = []
  common_objects = set(results_yt.keys())
  objects_yh = set(resutls_yh.keys())
  for common_object in common_objects:
    if common_object in objects_yh:
      zero_yt = np.zeros((512,512))
      zero_yh = np.zeros((512,512))
      for bbox in results_yt[common_object]:
        zero_yt[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
      for bbox in resutls_yh[common_object]:
        zero_yh[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
      c = np.sum((zero_yh * zero_yt)> 0)
      s = np.sum((zero_yh + zero_yt)> 0)
      iou_recall.append(c/s)
    else:
      iou_recall.append(0)
  iou_recall = np.mean(iou_recall) if len(iou_recall) > 0 else 0.0

  return {
      "common": iou_common,
      "precision": iou_precision,
      "recall": iou_recall
  }
