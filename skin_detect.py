import numpy as np
import scipy as sp
import cv2

class SkinDetect:

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

  @classmethod
  def texture_filter(self, img_gray, blur_size=5, limit=0.95):
    '''
    Finds rough textures, like cloth or leather, and returns 
    a mask representing this area.
    '''
    img_blur = cv2.blur(img_gray, (blur_size, blur_size))
    with np.errstate(divide='ignore', invalid='ignore'):
      mask = np.abs((img_gray/img_blur)) >= limit
    mask = np.array(mask * 255, np.uint8)
    return mask
    
  @classmethod
  def bgr_rule(self, img_bgr, lower=[40, 70, 120], upper=[240, 240, 240], gr_limit=20):
    '''
    Uses the BGR color model to filter out skin colors using
    various constraints. Returns a mask representing the are
    containing skin color.
    '''
    lower_bgr = np.array(lower, dtype="uint8")
    upper_bgr = np.array(upper, dtype="uint8")
    
    mask_1 = cv2.inRange(img_bgr, lower_bgr, upper_bgr)
    mask_2 = img_bgr[:,:,2] > img_bgr[:,:,1]
    mask_3 = img_bgr[:,:,2] > img_bgr[:,:,0]
    mask_4 = np.abs(img_bgr[:,:,2] - img_bgr[:,:,1]) > gr_limit
    return (mask_1 & mask_2 & mask_3 & mask_4) * 255
    
  @classmethod
  def hsv_rule(self, img_hsv, lower=[5, 60, 80], upper=[15, 165, 240]):
    '''
    Uses the HSV color model to filter out skin colors using
    various constraints. Returns a mask representing the are
    containing skin color.
    '''
    lower_hsv = np.array(lower, dtype="uint8")
    upper_hsv = np.array(upper, dtype="uint8")
    
    return cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    
  @classmethod
  def fill(self, mask, alpha):
    '''
    Uses connected components to fill in areas of the mask that are 
    larger than alpha percent of the mask size. Returns a new mask.
    '''
    inverted = cv2.bitwise_not(mask)
    
    height, width = mask.shape
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted)
    
    weighted = stats[labels, cv2.CC_STAT_AREA]
    
    filtered_black = weighted > alpha * height * width
    mask = (filtered_black & inverted) * 255
    
    return cv2.bitwise_not(mask)
    
  @classmethod
  def background_filter(self, img, alpha=0.6, beta=0.8):
    '''
    Perform canny edge detection to detect objects and create a mask 
    with the background removed. Returns a mask without the background.
    '''
    median = np.median(img)
    lower = int(max(0, (1.0 - beta) * median))
    upper = int(min(255, (1.0 - alpha) * median))
    
    mask = cv2.Canny(img, lower, upper)
    mask = cv2.dilate(mask, self.kernel, iterations=2)
    mask = cv2.erode(mask, self.kernel, iterations=1)
    mask = np.bitwise_not(self.fill(np.bitwise_not(self.fill(mask, 0.07)), 0.035))
    return mask
    
  @classmethod
  def contains_skin(self, img_bgr):
    '''
    Tries to detect skin in photographs by combining constraints on the 
    BGR and HSV color models, texture detection, and background detection 
    techniques. Returns the mask and True if skin is detected.
    '''
    # Scale the image to roughly 1500**2 pixels
    height, width, _ = img_bgr.shape
    scale = ((1500.0**2) / (height * width))**0.5
    
    img_bgr = cv2.resize(img_bgr, (0,0), fx=scale, fy=scale)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Compute masks and combine them to cover the common pixel areas
    mask = self.background_filter(img_gray)
    mask = cv2.bitwise_and(mask, self.texture_filter(img_gray))
    mask = cv2.bitwise_and(mask, self.hsv_rule(img_hsv))
    mask = cv2.bitwise_and(mask, self.bgr_rule(img_bgr))
    
    # Remove stray pixels and combine larger areas in the mask
    mask = cv2.erode(mask, self.kernel, iterations=1)
    mask = cv2.dilate(mask, self.kernel, iterations=1)
    
    # Remove smaller 'dots' from the mask 
    mask = np.bitwise_not(self.fill(np.bitwise_not(mask), 0.0001)) 
    
    # What's left in the mask is hopefully skin
    height, width, _ = img_bgr.shape
    weight = cv2.countNonZero(mask) / float(height * width)
    
    return mask, weight > 0
  
