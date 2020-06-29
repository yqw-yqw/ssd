from ssd import SSD
from PIL import Image
import torch

ssd = SSD()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        with torch.no_grad():
          r_image = ssd.detect_image(image)
          r_image.show()
          r_image.save('G:/pytorch程序/shiyanxuyao/predict/test5.jpg')
