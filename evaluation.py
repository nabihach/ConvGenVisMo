from PIL import Image
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import numpy as np
from brisque import BRISQUE
import sewar
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

class ConGenVismoEval():
  def __init__(self):
    """Loads the respective models required for the evaluation.
    """
    self.__clip = SentenceTransformer('clip-ViT-B-32')
    self.__brisque = BRISQUE(url=False)
    self.__dtr_processor = processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    self.__dtr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
  
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
