from PIL import Image
from matplotlib.pyplot import figure, subplot
from matplotlib.pyplot import show
from matplotlib.pyplot import imshow
from numpy import *
from scipy.cluster.vq import *
from scipy.misc import imresize

steps=1000
im =array(Image.open('IMGP8080.JPG'))

dx=int(im.shape[0]/steps)
dy=int(im.shape[1]/steps)

features=[]
for x in range(steps):
    for y in range(steps):
        R=mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,0])
        G=mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,1])
        B=mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,2])
        features.append([R,G,B])
features=array(features,'f')

centroids,variance=kmeans(features,2)
code, distance =vq(features,centroids)

codeim=code.reshape(steps,steps)
codeim=imresize(codeim,im.shape[:2],interp='nearest')

figure
subplot(1,2,1)
imshow(im)
subplot(1,2,2)
imshow(codeim)
show()

