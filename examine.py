import cv2
import matplotlib.pyplot as plt
import numpy as np

img2 = cv2.imread(".\\data\\occlusion_00\\rgb\\0001.png")
plt.figure(1)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.show()

img = cv2.imread(".\\data\\occlusion_00\\stem_id_mask\\0001.png", cv2.IMREAD_GRAYSCALE)
print(np.max(img))
print(np.min(img))
plt.figure(2)
plt.imshow(img/np.max(img),cmap='gray')
plt.show()