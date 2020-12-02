class EZWDecoder:
  def __init__(self, M, N, start_thres, wavelet):
    img = np.zeros((M, N))
    self.wavelet = wavelet

    LL = np.copy(img)
    subs = []
    for i in range(1):
      LL, subbands = pywt.dwt2(LL, wavelet)
      subs.append(subbands)
    
    coeffs = LL
    for subbands in reversed(subs):
      coeffs = np.concatenate((
          np.concatenate((coeffs, subbands[0]), axis=1), 
          numpy.concatenate((subbands[1], subbands[2]), axis=1)), 
          axis=0)
    
    self.LLshape = LL.shape
    self.coeffs = coeffs
    self.thresh = start_thres
    self.processed = []

    for i in range(LL.shape[0]):
      for j in range(LL.shape[1]):
        self.tree.append(EZWNode(self.coeffs, (i,j), True, self.LLshape))

  def getImage(self):
    return pywt.idwt2( (self.coeffs[0:LLshape[0], 0:LLshape[1]], (self.coeffs[LLshape[0]:self.coeffs.shape[0], 0:LLshape[1]], self.coeffs[0:LLshape[0], LLshape[1]:self.coeffs.shape[1]], self.coeffs[LLshape[0]:self.coeffs.shape[0], LLshape[1]:self.coeffs.shape[1]])), self.wavelet)

  def dominant_pass(self, code_list):
    q = []
    for parent in self.tree:
      q.append(parent)

    for code in code_list:
      if len(q) == 0:
        break
      node = q.pop(0)
      if code != "T":
        for child in node.children:
          q.append(child)
      if code == "P" or code == "N":
        node.value = (1 if code == "P" else -1) * self.T
        self._fill_coeff(node)
        self.processed.append(node)

  def secondary_pass(self, bitarr):
    if len(bitarr) != len(self.processed):
      bitarr = bitarr[:len(self.processed)]
    for bit, node in zip(bitarr, self.processed):
      if bit:
        node.value += (1 if node.value > 0 else -1) * self.thresh // 2
        self._fill_coeff(node)

    self.thresh //= 2

  def _fill_coeff(self, node):
    self.coeffs[node.loc] = node.value
