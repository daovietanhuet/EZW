import pywt
import numpy as np
import matplotlib.pyplot as plt
import cv2

class EZWNode:
  def __init__(self, coeffs, loc, isLL, LLshape):
    self.children = []
    self.symbol = ""
    self.value = coeffs[loc]
    # print(loc)
    
    i,j = loc
    
    if not isLL:
      child_locs = [(2*i, 2*j), (2*i, 2*j + 1), (2*i + 1, 2*j), (2*i + 1, 2*j + 1)]
    else:
      child_locs = [(2*i + LLshape[0], j), (i, 2*j + LLshape[1]), (2*i + LLshape[0], 2*j + LLshape[1])]
    
    for cloc in child_locs:
      if cloc[0] >= coeffs.shape[0] or cloc[1] >= coeffs.shape[1]:
        continue
      else:
        self.children.append(EZWNode(coeffs, cloc, False, LLshape))
  
  def set_symbol(self, threshold):
    for child in self.children:
      child.set_symbol(threshold)

    if abs(self.value) >= threshold:
      self.symbol = "P" if self.value > 0 else "N"
    else:
      self.symbol = "Z" if any([child.symbol != "T" for child in self.children]) else "T"

class ZeroTreeEncoder:
  def __init__(self, image, wavelet, level):
    # Transform
    LL = np.copy(image)
    subs = []
    for i in range(level):
      LL, subbands = pywt.dwt2(LL, wavelet)
      subs.append(subbands)
    
    coeffs = LL
    for subbands in reversed(subs):
      coeffs = np.concatenate((
          np.concatenate((coeffs, subbands[0]), axis=1), 
          numpy.concatenate((subbands[1], subbands[2]), axis=1)), 
          axis=0)

    #Quantize 
    coeffs = np.sign(coeffs) * np.floor(np.abs(coeffs))
      
    self.coeffs = coeffs
    self.LLshape = LL.shape
    self.tree = []
    self.thresh = np.power(2, np.floor(np.log2(np.max(np.abs(coeffs)))))
    self.second_list = []

    #Build coefficient tree
    for i in range(LL.shape[0]):
      for j in range(LL.shape[1]):
        self.tree.append(EZWNode(self.coeffs, (i,j), True, self.LLshape))

  def dominant_pass(self):
    q = []
    sec = []
    for parent in self.tree:
      parent.set_symbol(self.thresh)
      q.append(parent)

    codes = []
    while len(q) != 0:
      node = q.pop(0)
      codes.append(node.symbol)

      if node.symbol != "T":
        for child in node.children:
          q.append(child)

      if node.symbol == "P" or node.symbol == "N":
        sec.append(node.value)
        node.value = 0
    
    self.second_list = np.abs(np.array(sec))

    return codes

  def second_pass(self):
    bits = []

    middle = self.thresh // 2
    for i, coeff in enumerate(self.second_list):
      if coeff - self.thresh >= 0:
        self.second_list[i] -= self.thresh
      bits.append(self.second_list[i] >= middle)
    return bits

    
if __name__ == "__main__":
  img = cv2.imread('im.png', 0)
  ezw = ZeroTreeEncoder(img, 'haar', 1)
  D = ezw.dominant_pass()
  S = ezw.second_pass()
