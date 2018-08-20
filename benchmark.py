import cv2
import os
from skin_detect import SkinDetect
from matplotlib import pyplot as plt
from random import shuffle

human_path = './images/human/'
not_human_path = './images/not_human/'

humans = set(os.listdir(human_path))
not_humans = set(os.listdir(not_human_path))
images = sorted([fil for fil in (humans | not_humans) if fil.endswith('.jpg')])
shuffle(images)

positive = 0
negative = 0
errors_positive = 0
errors_negative = 0

for i, fil in enumerate((images)):
  img_bgr = None
  if fil in humans:
    img_bgr = cv2.imread(human_path + fil, cv2.IMREAD_COLOR)
  elif fil in not_humans:
    img_bgr = cv2.imread(not_human_path + fil, cv2.IMREAD_COLOR)
  
  mask, detected = SkinDetect.contains_skin(img_bgr)
      
  if fil in humans:
    negative += 1
    if not detected:
      errors_negative += 1
  else:
    positive += 1
    if detected:
      errors_positive += 1
  
  print("t: %.3f\tp: %.3f\tn: %.3f\t%d/%d\t\t%s" % 
        (100*(errors_positive+errors_negative)/(positive+negative), 
         100*(errors_positive)/(max(positive, 1)), 
         100*(errors_negative)/(max(negative, 1)),
         i+1, len(images), fil))
    
print('Cnt:', len(images))
print('Hmn:', len(humans))
print('All:', errors_positive + errors_negative)
print('Pos:', errors_positive)
print('Neg:', errors_negative)

