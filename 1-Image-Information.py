#!/usr/bin/env python
# coding: utf-8

# # Image Information
# 
# The main objectives of this module are:
# 
# * Manipulate an image with Python, scikit-image and numpy.
# * Process images at the pixel level.
# * Compute and understand image histograms.
# * Understand lossless compression & reconstruction.
# * Understand the co-occurrence matrix.
# * Use different colour representations.

# ## 1. Read & write an image
# 
# In this exercise, we will simply open an image file, display it, and save a copy. 
# 
# **Use the [scikit-image io](https://scikit-image.org/docs/dev/api/skimage.io.html) module to open, show & save a copy of the "camera.jpg" image.**
# 
# *Note: we use the **%matplotlib inline** command to display the image in the notebook. It would not be necessary if you execute the code in the terminal or in a standard IDE like PyCharm.*

# In[69]:


#usefull fonction 
import numpy as np
def arrayHist(x):
    hist,bins = np.histogram(x.flatten(),range(257))  # histogram is computed on a 1D distribution --> flatten()
    return hist,bins
def arrayHistNormalized(x):
    hist,bins = np.histogram(x.flatten(),range(257), density=True)  # histogram is computed on a 1D distribution --> flatten()
    return hist, bins


# In[436]:


from skimage.io import imread,imsave,imshow

get_ipython().run_line_magic('matplotlib', 'notebook')

img_array = imread("camera.jpg")
imshow(img_array)
imsave("camera_copy.jpg",img_array)
## -- Your code here -- ##


# When you open an image with scikit-image, it is stored as a Numpy [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) object. Numpy arrays are objects which can be easily manipulated for numerical computations.
# 
# **Using *ndarray* methods & attributes, answer the following questions about the "camera" image:**
# 
# 1. What is the shape of the image? (width & height)
# 1. What is the minimum pixel value? What is the maximum pixel value?
# 1. What is the data type for each pixel?
# 1. Show only a 100x100 pixels window taken at the center of the image.

# In[71]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')

print (img_array.shape)
print (min(img_array.flatten()))
print (max(img_array.flatten()))
img_array = imread("camera.jpg")


img_cropped = np.zeros([100,100])
print (img_cropped.shape)

center1 = int(img_array.shape[0]/2)
center2 = int(img_array.shape[1]/2)
print (center1, center2)
img_cropped = img_array[center1-50:center1+50, center2-50:center2+50]

imshow(img_cropped)
        

## -- Your code here -- ##


# ## 2. Image histograms
# 
# * Compute and plot the **histogram** and the **normalized histogram** of the example cameraman image given below.
# 
# You can use the [pyplot module](https://matplotlib.org/api/pyplot_api.html) from matplotlib to display plots & histograms.

# In[217]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

## -- Your code here -- ##
img_array = imread("camera.jpg")

hist, bins = arrayHist(img_array)
print (hist)
plt.figure()
plt.bar(bins[:-1],hist)
plt.show()

plt.figure()
hist, bins = arrayHistNormalized(img_array)
plt.bar(bins[:-1],hist)
plt.show()


# Can you use the histogram to answer the following questions ? (you don't need to code anything here)
# 
# 1. What is the average gray value of the cameraman ?
# 1. What is the average gray value of the sky ?
# 1. Is there more 'cameraman' pixels than 'sky' pixels ?

# Compute and plot the **cumulated histogram**.

# In[78]:


get_ipython().run_line_magic('matplotlib', 'inline')

hist, bins = arrayHistNormalized(img_array)
cumul_hist = np.cumsum(hist)
plt.plot(cumul_hist,label='cumul.',color='k')

## -- Your code here -- ##


# ## 3. Image entropy
# 
# The "entropy" of a signal, in information theory, can generally be interpreted as the "number of bits required to encode the signal". It is a measure of the "amount of information" contained in the signal. Intuitively, a signal with a very narrow distribution (all values are close to each other) will have a very low entropy, while a signal with a wide distribution (the values are evenly distributed) will have a higher entropy.
# 
# 1. Compute the image entropy of the cameraman image. The image entropy is given by $e = - \sum_{g=0}^N p(g) \log_2(p(g))$ where $p(g)$ is the probability that a pixel has the grayscale value g, and N is the number of possible grayscale values. Note that p(g) is directly given by the normalized histogram.
# 1. What is the entropy of a shuffled version of the cameraman ?

# In[74]:


def calcEntropy(img):
    entropy = 0
    for x in range (len(img)):
        if (img[x] > 0 ):
            entropy = entropy + img[x] * np.log2(img[x])
    
    return entropy * -1


# In[80]:


proba,bins = arrayHistNormalized(img_array)
print (len(proba))
print(np.sum(proba)) #should give one
print (calcEntropy(proba))
#it needs 4.89 bits in average to store each pixel. So the image could a lot compressed





shuffled_image = img_array
np.random.shuffle(shuffled_image)
proba_shuff, bins = arrayHistNormalized(img_array)
print (calcEntropy(proba_shuff))
#entropy is the same

# -- Your code here -- #


# ## 4. Image compression
# 
# Using the code below as a starting point:
# 
# * **Decompose an image** by recursively subsampling its dimensions and computing the remainders, such that each level of recursion performs the following operation:
# 
# <img src='./PyramidCompression.png' width='75%'/>

# **Compute how the image entropy evolves** with regards to the level of decomposition

# In[88]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.io import imread

# Modify this method:
def split(im):
    if im.shape[0] > 1:
        a = im[0:-1:2,0:-1:2]
        b = im[0:-1:2,1::2]-a
        c = im[1::2,0:-1:2]-a
        d = im[1::2,1::2]-a
        R = np.vstack((np.hstack((split(a),b)),np.hstack((c,d))))
        hist,bins = np.histogram(R.flatten(),range(-256,257), density=True)  
        print ('entropy',calcEntropy(hist))
    else:
        R = im
    return R
im = imread('camera.jpg')
print ('im', type(im))
im = im.astype(np.int16) # cast the camera image as a signed integer to avoid overflow
s = split(im)

plt.figure(figsize=(12,12))
# interpolation='nearest' -> don't try to interpolate values between pixels if the size of the display is different from the size of the image
# cmap=cm.gray -> display in grayscale
# vmin=-255 -> set "black" as -255
# vmax=255 -> set "white" as 255
plt.imshow(s,interpolation='nearest',cmap=cm.gray,vmin=-255, vmax=255)
plt.colorbar()




hist,bins = np.histogram(s.flatten(),range(-256,257), density=True)  
print('proba',np.sum(hist))


plt.figure()
plt.bar(bins[:-1],hist)
plt.show()


print (calcEntropy(hist))


# **Rebuild the original image** from the pyramid (allowing the selection the level of recursion)

# In[84]:


#Reconstruit

def reconstruct(im):
    print(im.shape)
    if (im.shape[0]%2!=0 or im.shape[1]%2!=0 ): return im
        
        
        
    midx = im.shape[1]//2
    midy = im.shape[0]//2

    a=reconstruct(im[:midy, :midx])
    b=im[:midy, midx:]
    c=im[midy:, :midx]
    d=im[midy:, midx:]
        
    imstack = im.copy()
    imstack[::2, ::2] = a
    imstack[::2, 1::2] = b+a
    imstack[1::2, ::2] = c+a
    imstack[1::2, 1::2] = d+a
    return imstack



im = (imread("camera.jpg")).astype(np.int16) # cast the camera image as a signed integer to avoid overflow
s = split(im)
plt.figure(figsize=(12,12))
plt.imshow(s,interpolation='nearest',cmap=cm.gray,vmin=-255, vmax=255)
plt.colorbar()

im_recon = reconstruct(s)
plt.figure()
plt.imshow(im_recon, cmap= cm.gray)
plt.show()


# ## 5. Co-occurrence matrix
# 
# While the histogram of an image is independent of the position of the pixels, the co-occurrence matrix gives us information about their spatial distribution.
# 
# A co-occurrence matrix is computed for a given displacement, looking at the pair of values spatially separated by that displacement. The co-occurrence matrix is a square matrix, its size given by the number of possible values that a pixels can take in the image.
# 
# 1. Compute de [cooccurrence matrix](https://en.wikipedia.org/wiki/Co-occurrence_matrix) for a chosen displacement $(\Delta x,\Delta y)$ (see [greycomatrix](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.greycomatrix) in scikit-image)
# 1. What is the entropy of the cooccurrence matrix ?
# 1. How does this entropy evolve if we increase the displacement ?

# In[165]:


print(img_array[10,5])

#!!!! img_array[i,j] = img_array[y,x] !!!! Il faut inverser x et y car i représente les lignes et j les colonnes!
def log_nz(im1):
    im2 = im1.copy()
    im2[im1==0] = 0.5
    return np.log(im2)


# In[167]:


from skimage.feature import greycomatrix
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')

img_array = imread("camera.jpg")
dx = 5
dy = 10

# img_array = np.zeros((512,512)) #For C show only value  at 0,0 512,512 
# img_array[:,0:-1:10]= 255
# plt.imshow(img_array)

C = np.zeros((256,256)) #number of pixel value that a pixel can take
print(img_array.shape)
for x in range (img_array.shape[0]-dx):
    for y in range(img_array.shape[1]-dy):
#         if (x+dx < img_array.shape[0]-1 and y+dy < img_array.shape[1]-1):
            i = int(img_array[x,y])
            j = int(img_array[x+dx,y+dy])
            C[i,j] += 1
    
    




from sklearn.preprocessing import MinMaxScaler
import sys
np.set_printoptions(threshold=sys.maxsize)
plt.figure()
plt.imshow(log_nz(C), cmap = cm.jet)
print(C[0:20,10:20])
# -- Your code here -- #

# maxC = max(C.flatten())

# C = ((C/maxC)*255).astype(np.int8)
# plt.figure()
# plt.imshow(C)
         
         


# In[181]:


from skimage.feature import greycomatrix
img_array = imread("camera.jpg")
distance = [11.18] #pour avoir le même sqrt(5² + 10²) et angle de 60° pour avoir dx = 5 et dy = 10
angles =[np.pi/3]

co_matrix = greycomatrix(img_array,distance,angles).astype('float')
print(co_matrix.shape)
plt.figure()
plt.imshow(log_nz(co_matrix[:,:,0,0]), cmap = cm.jet)


# ## 6. Colour representations
# 
# A colour image is typically encoded with three channels: Red, Green and Blue. In the example below, we open the *immunohistochemistry()* example image and split it into the three channels, which we display: 

# In[10]:


from skimage.data import immunohistochemistry

im = immunohistochemistry() # scikit-image method to load the example image
print(im.shape,im.dtype)
r = im[:,:,0]
g = im[:,:,1]
b = im[:,:,2]

plt.gray() # Use grayscale by default on 1-channel images, so you don't have to add cmap=plt.cm.gray everytime

plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.imshow(im)
plt.title('RGB')
plt.subplot(2,2,2)
plt.imshow(r)
plt.title('Red')
plt.subplot(2,2,3)
plt.imshow(g)
plt.title('Green')
plt.subplot(2,2,4)
plt.imshow(b)
plt.title('Blue')
plt.show()


# 1. Compute & show the color histograms
# 1. Convert the image to the HSV color space & compute the HSV histograms. [See the skimage documentation for reference on color transformation](http://scikit-image.org/docs/dev/api/skimage.color.html#rgb2hsv)
# 1. Find a method to isolate the brown cells in the immunohistochemistry image
#     1. In the RGB space
#     1. In the HSV space

# # Beginning of the project : Analazing the problem

# ### Plot the image to add the watermark

# In[405]:


from skimage.io import imread,imsave,imshow
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
im_water = imread('etretat.jpg')
plt.imshow(im_water)


# ### Analyze of the watermark

# In[406]:


#First plot ulb logo
get_ipython().run_line_magic('matplotlib', 'notebook')

im_ulb = imread('watermark.png')
plt.figure()
plt.imshow(im_ulb)
print(im_ulb.shape) # Output -> (85,219)

hist,bins = np.histogram(im_ulb.flatten(),range(257))  # histogram is computed on a 1D distribution --> flatten()
#As we can see it is a binary image
#To make sure of it, let's plot an histogram
plt.figure()
hist, bins = arrayHistNormalized(im_ulb)
plt.bar(bins[:-1],hist)
plt.show()
#We cann see it's binary. There is more white pixel than black pixel since the peak is higher for the value 255.


# ### There are here all the variable that needs to be initialized if you don't run all the cell from up to down. I will also initialize again theses variables in some block when I explain the code. Good lecture ! 

# In[525]:


im_ulb = imread('watermark.png')
im_water = imread('etretat.jpg')
midx_water = im_water.shape[1]//2
midy_water = im_water.shape[0]//2
height_ulb = im_ulb.shape[0]
width_ulb = im_ulb.shape[1]
h_up = height_ulb//2
h_down = height_ulb//2
if (h_up + h_down < height_ulb):
    h_up +=1
    
w_left = width_ulb//2
w_right = width_ulb//2

if (w_left + w_right < width_ulb):
    w_left +=1

    
def save_im(im, filepath):
    imsave(filepath, im)
    
def display_im(im, size, title):
    plt.figure(figsize=size)
    plt.title(title)
    plt.imshow(im)
    plt.show()
    


# # The code begins ! 
# 
# #  Step 1- Add the white pixels from the watermark somewhere in the photograph +Save the resulting image as an image file & display it in the notebook 

# In[563]:


#First step, let's simply integrate the ulb to the image with no transparency 
im_ulb = imread('watermark.png')
im_water = imread('etretat.jpg')

# plt.figure(figsize=(12,12))
# plt.imshow(im_water)

midx_water = im_water.shape[1]//2
midy_water = im_water.shape[0]//2

height_ulb = im_ulb.shape[0]
width_ulb = im_ulb.shape[1]
print('midx_water',midx_water,'midy_water', midy_water)
print('height', height_ulb, 'width', width_ulb)


# for x in range(3):
#     im_water[midy_water-height_ulb//2:midy_water+height_ulb//2, midx_water-width_ulb//2:midx_water+width_ulb//2,0] = im_ulb

#This line doesn't work all the time... the issue is that one of the dimension is an odd number, it won't work
#In this case, it doesn't work ! as 85 and 219 are odd number, 1 pixel in lost with the division for each direction. Let's add
# +1 to each dimension ! 

for x in range(3):
    im_water[midy_water-height_ulb//2:midy_water+height_ulb//2+1, midx_water-width_ulb//2:midx_water+width_ulb//2+1,x] = im_ulb

#Okay that's working ! But it's not an elegant way to it. If the resolution of the images changes to a even number, it will 
#crash ! 
#Let's build a solution for every shape of image, solution is the next block ! 


    

display_im(im_water, (8,8), "step 1")
save_im(im_water, "step_1.jpg")


# # Step 2 - Add the white pixels from the watermark somewhere in the photograph +Save the resulting image as an image file & display it in the notebook 

# In[409]:


#First step, let's simply integrate the ulb to the image with no transparency 
im_ulb = imread('watermark.png')
im_water = imread('etretat.jpg')
h_up = height_ulb//2
h_down = height_ulb//2

if (h_up + h_down < height_ulb):
    h_up +=1
    
w_left = width_ulb//2
w_right = width_ulb//2

if (w_left + w_right < width_ulb):
    w_left +=1

for x in range(3):
    im_water[midy_water-h_down:midy_water+h_up, midx_water-w_left:midx_water+w_right,x] = im_ulb

#Okay that's cool, it's working, but we got the black pixel... It's annoying ! Let's put a condition to not have black pixel!

display_im(im_water, (8,8), "step 2")
save_im(im_water, 'step_2.jpg')


# # Step 3 - Add the white pixels from the watermark somewhere in the photograph +Save the resulting image as an image file & display it in the notebook 

# In[410]:


#First step, let's simply integrate the ulb to the image with no transparency 
im_ulb = imread('watermark.png')
im_water = imread('etretat.jpg')
color_mark = 0 #choose the color here (255 for white and 0 for black)
for y in range(midy_water-h_down,midy_water+h_up,1):
    for x in range(midx_water-w_left,midx_water+w_right,1):
        pixel_value = im_ulb[y-(midy_water-h_down),x-(midx_water-w_left)]
        if (pixel_value == 255):
            for z in range(3):
                im_water[y,x,z] =  color_mark
                
display_im(im_water, (8,8), "step 3")
save_im(im_water, 'step_3.jpg')
#Okay that's cool ! It seems to be working pretty well ! 
#I would like it to be more transparent ! 


# # Step 4 : Add transparency effect to the watermark

# In[589]:


im_ulb = imread('watermark.png')
im_water = imread('etretat.jpg')

transparency_intensity =150 #choose the transparency here (255 is opaque, 0 is invisible)


for y in range(midy_water-h_down,midy_water+h_up,1):
    for x in range(midx_water-w_left,midx_water+w_right,1):
        pixel_value = im_ulb[y-(midy_water-h_down),x-(midx_water-w_left)]
        if (pixel_value == 255):
            for z in range(3):
                total_pixel = im_water[y,x,z] + transparency_intensity
                if (total_pixel > 255):
                    total_pixel = 255
                im_water[y,x,z] =  total_pixel
                
display_im(im_water, (8,8), "step 4")
save_im(im_water, 'step_4.jpg')


# # Step 5 - Add an option to choose the watermark location

# In[591]:


def onclick(event):
    #tx = 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata)
    #text.set_text(tx)
   
    y_chosen =int(event.ydata)
    x_chosen =int(event.xdata)
    x_chosen, y_chosen = correct_location(x_chosen, y_chosen)
    im_water = imread('etretat.jpg')
    im_ulb = imread('watermark.png')
    
    transparency_intensity =110 #choose the transparency here (255 is opaque, 0 is invisible)
    
    for y in range(y_chosen-h_down,y_chosen+h_up,1):
        for x in range(x_chosen-w_left,x_chosen+w_right,1):
            pixel_value = im_ulb[y-(y_chosen-h_down),x-(x_chosen-w_left)]
            if (pixel_value == 255):
                for z in range(3):
                    total_pixel = im_water[y,x,z] + transparency_intensity
                    if (total_pixel > 255):
                        total_pixel = 255
                    im_water[y,x,z] =  total_pixel
    ax.imshow(im_water)
    save_im(im_water, 'step_5.jpg')
    

def correct_location(x,y):
   

    if (x  > width_ulb/2 +1 and x <  (im_water.shape[1]- (width_ulb/2)-1) 
        and  y > height_ulb/2 +1 and y <  (im_water.shape[0]- (height_ulb/2)-1) ):
        return x,y
    if (x < ((width_ulb/2) +1)):
        x = ((width_ulb//2) +1)
    if (x > im_water.shape[1]- (width_ulb/2)-1):
        x = im_water.shape[1]- (width_ulb//2)-1 
    if (y < height_ulb/2 +1):
        y = height_ulb//2 +1 
    if (y > im_water.shape[0]- (height_ulb/2)-1):
        y = im_water.shape[0]- (height_ulb//2)-1 
    return x,y
        
        
    
# def correct_location(x,y):
#     if ((y_chosen < im_water.shape[0] - height_ulb) and (y_chosen > width_ulb) and 
#         (x_chosen < im_water.shape[1] - width_ulb) and (x_chosen > width_ulb)):
#         return x,y
#     else:
#         return 250,250


# In[592]:


#Simply click on the image and it appears where you want ! 
im_water = imread('etretat.jpg')
fig = plt.figure(figsize=[15,8])
ax= fig.add_subplot(111)
ax.imshow(im_water)
ax.set_title("Click on the picture to add the watermark!")
text=ax.text(0,0, "", va="bottom", ha="left")
cid = fig.canvas.mpl_connect('button_press_event', onclick)








# # Step 6 : Make a black transparency watermark instead of a white one

# In[627]:


im_ulb = imread('watermark.png')
im_water = imread('etretat.jpg')

transparency_intensity =70 #choose the transparency here (255 is opaque, 0 is invisible)


for y in range(midy_water-h_down,midy_water+h_up,1):
    for x in range(midx_water-w_left,midx_water+w_right,1):
        pixel_value = im_ulb[y-(midy_water-h_down),x-(midx_water-w_left)]
        if (pixel_value == 255):
            for z in range(3):
                total_pixel =  im_water[y,x,z] - transparency_intensity
                if (total_pixel <0 ):
                    total_pixel = 0
                im_water[y,x,z] =  total_pixel
                
display_im(im_water, (8,8), "step 6")
save_im(im_water, 'step_6.jpg')


# # Let's check the progression !  

# In[593]:


def display_2_subplot(im1,im2,t1,t2):
    plt.figure(figsize=[15,8])
    plt.subplot(1,2,1).set_title(t1)
    plt.imshow(im1)
    plt.subplot(1,2,2).set_title(t2)
    plt.imshow(im2)

if (1==2):
    print(1)
else:
    print(2)


# In[628]:


number_step = 6
i = 1

    
while i < number_step+1:
    x = i
    if (x + 1< number_step+1):
        display_2_subplot(imread("step_"+str(x)+".jpg"), imread("step_"+str(x+1)+".jpg"), 'step '+ str(x),'step '+ str(x +1))
        i+=2
    else:
        display_im(imread("step_"+str(x)+".jpg"),(8,8), "step "+str(x))
        i+=1
    
    


# # Coding Project - Watermark
# 
# Write code to automatically add a watermark to a photograph.
# 
# <img src='./ex_wm.jpg' width="500px" />
# 
# ## Main requirements
# 
# The minimum requirements are to:
# * Add the white pixels from the watermark somewhere in the photograph.
# * Save the resulting image as an image file & display it in the notebook
# 
# You may use the *watermark.png* file available in the GitHub repository, or choose/create your own.
# 
# ## Additional requirements
# 
# For extra points, you may add some possible improvements (note: this is not an exhaustive list, use your imagination!)
# 
# * Add an option to choose the watermark location
# * Add transparency effect to the watermark
# * Determine if the watermark should be dark or light based on the luminosity of the image
# * ...
# 

# In[12]:



# -- Your code here -- #

