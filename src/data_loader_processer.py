import cv2 
import os 

#path to snippets
snippets_path = 'data/snippets/'
#make sure to check ends with the correct file extension
image_files = [f for f in os.listdir(snippets_path) if f.endswith('.png')]

images = []

for file in image_files:
    img_path = os.path.join(snippets_path, file)
    img = cv2.imread(snippets_path)
    images.append((file, img))
