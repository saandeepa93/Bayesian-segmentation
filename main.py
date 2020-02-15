import os
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

curr_dir = os.getcwd()+"\Images"

coord_dict = {}
coord_dict['img1'] = []

def click_event(event, x, y, flags, img):
    if event == cv2.EVENT_LBUTTONDOWN:
        lst = [x,y]
        print(lst)
        print(img[x][y])
        coord_dict['img1'].append(lst)
       
def Lst_to_Array(lst):
    lst = np.array(lst).astype(float)
    return lst

def Gaussian_func(mean, covar):
    def memoise(x):
        const =  (1/(np.sqrt(np.linalg.det(covar))*np.sqrt(2*np.pi)))
        exp = np.exp((-0.5)*(np.square(x-mean)/np.linalg.det(covar)))
        return const*exp
    return memoise

def Swap(arr):
    temp = arr[2]
    arr[2] = arr[0]
    arr[0] = temp
    return arr

'''
Requires manually changing the filename for each run
Change the filename for each type of data being collected
'''
def Collect_data():
    sky_lst = []
    for filename in os.listdir(curr_dir):
        if filename == "img9.png":
    #    if filename.endswith(".png"):
            print(filename)
            img_read = cv2.imread(curr_dir+"\\"+filename)
            cv2.imshow('image',img_read)
            cv2.setMouseCallback("image",click_event,img_read)
            cv2.waitKey()
            cv2.destroyAllWindows
            
            xmin = coord_dict['img1'][-2][0]
            ymin = coord_dict['img1'][-2][1]
            
            xmax = coord_dict['img1'][-1][0]
            ymax = coord_dict['img1'][-1][1]
            print("\n dict values")
            print(xmin,ymin)
            print(xmax,ymax)
            result = img_read[ymin:ymax,xmin:xmax,:]
            
            sky_lst = result if len(sky_lst) == 0 else np.vstack((sky_lst,result))
            
            print("Skylist dimensions",sky_lst.shape)
            
#            temp = sky_lst[:,:,2]
#            sky_lst[:,:,2] = sky_lst[:,:,0]
#            sky_lst[:,:,0] = temp
            
            sky_lst = np.array(sky_lst)
            width, height, dims = sky_lst.shape  
            
            sky_lst = np.reshape(sky_lst,(width*height,dims))
            
            
            
            with open('Sky.csv','a',newline = "") as handle:
                writer = csv.writer(handle)
                writer.writerows(sky_lst)  
                
            
                
                
Collect_data()

'''
comment the below code if you run collect_data method
'''        

with open('Bird.csv','r') as handler:
    bird_lst = list(csv.reader(handler))

with open('Sky.csv','r') as handler:
    sky_lst = list(csv.reader(handler))

with open('Cloud.csv','r') as handler:
    cloud_lst = list(csv.reader(handler))
    
bird_arr = Lst_to_Array(bird_lst).astype(np.uint8)
sky_arr = Lst_to_Array(sky_lst).astype(np.uint8)
cloud_arr = Lst_to_Array(cloud_lst).astype(np.uint8)

sky_img = np.reshape(sky_arr[:120000],(600, 200, 3))
cv2.imshow('image',sky_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

bird_img = np.reshape(bird_arr[:15000],(150,100 , 3))
cv2.imshow('image',bird_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cloud_img = np.reshape(cloud_arr[:39000],(300,130, 3))
cv2.imshow('image',cloud_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


bird_arr = bird_arr[:15000]
sky_arr = sky_arr[:15000]
cloud_arr = cloud_arr[:15000]

bird_mean, bird_cov = np.mean(bird_arr,axis = 0), np.cov(bird_arr.transpose())
sky_mean, sky_cov = np.mean(sky_arr,axis = 0), np.cov(sky_arr.transpose())
cloud_mean, cloud_cov = np.mean(cloud_arr, axis =0), np.cov(cloud_arr.transpose())

bird_mean = Swap(bird_mean)
sky_mean = Swap(sky_mean)
cloud_mean = Swap(cloud_mean)

bird_cov = bird_cov.transpose()
sky_cov = sky_cov.transpose()
cloud_cov = cloud_cov.transpose()


#Edited here
bird_mean = np.reshape(bird_mean,(bird_mean.shape[0],1))
sky_mean = np.reshape(sky_mean,(sky_mean.shape[0],1))
cloud_mean = np.reshape(cloud_mean,(cloud_mean.shape[0],1))

def Plot_normal(x_bird,x_sky,x_cloud,mu_bird,mu_sky,mu_cloud,sigma_bird,sigma_sky,sigma_cloud):
    temp = np.array([i for i in range(-1000, 1000)])
    
#    y_pdf_bird_r = norm.pdf(temp, mu_bird[0], sigma_bird[0,0])
    y_pdf_bird_g = norm.pdf(temp, mu_bird[1], sigma_bird[0,0])
#    y_pdf_bird_b = norm.pdf(temp, mu_bird[2], sigma_bird[0,0])
#    
#    plt.plot(temp,y_pdf_bird_r,label='Bird Normal Distribution',color = 'r')
    plt.plot(temp,y_pdf_bird_g,label='Bird Normal Distribution', color = 'r')
#    plt.plot(temp,y_pdf_bird_b,label='Bird Normal Distribution', color = 'r')
    
#    plt.axvline(x=mu_bird[0],color='r')
    plt.axvline(x=mu_bird[1],color='r')
#    plt.axvline(x=mu_bird[2],color='r')
    
#    plt.axvline(x=mu_sky[0],color='g')
    plt.axvline(x=mu_sky[1],color='g')
#    plt.axvline(x=mu_sky[2],color='g')
    
#    plt.axvline(x=mu_cloud[0],color='b')
    plt.axvline(x=mu_cloud[1],color='b')
#    plt.axvline(x=mu_cloud[2],color='b')
    
#    y_pdf_sky_r = norm.pdf(temp, mu_sky[0], sigma_sky[1,1])
    y_pdf_sky_g = norm.pdf(temp, mu_sky[1], sigma_sky[1,1])
#    y_pdf_sky_b = norm.pdf(temp, mu_sky[2], sigma_sky[1,1])
#    
#    plt.plot(temp,y_pdf_sky_r,label='Bird Normal Distribution',color = 'g')
#    plt.plot(temp,y_pdf_sky_g,label='Bird Normal Distribution', color = 'g')
#    plt.plot(temp,y_pdf_sky_b,label='Bird Normal Distribution', color = 'g')
    
#    y_pdf_cloud_r = norm.pdf(temp, mu_cloud[0], sigma_cloud[2,2])
    y_pdf_cloud_g = norm.pdf(temp, mu_cloud[1], sigma_cloud[2,2])
#    y_pdf_cloud_b = norm.pdf(temp, mu_cloud[2], sigma_cloud[2,2])
#    
#    plt.plot(temp,y_pdf_cloud_r,label='Bird Normal Distribution',color = 'b')
    plt.plot(temp,y_pdf_cloud_g,label='Bird Normal Distribution', color = 'b')
#    plt.plot(temp,y_pdf_cloud_b,label='Bird Normal Distribution', color = 'b')
#    
    
    plt.set_xlim([-255,255])
    plt.show

bird_mean_avg = np.mean(bird_mean,axis=1)

Plot_normal(bird_arr[:15000],sky_arr[:15000],cloud_arr[:15000]
            ,bird_mean,sky_mean,cloud_mean
            ,bird_cov,sky_cov,cloud_cov)




print(bird_arr.shape)
#Edited here


test_img = cv2.imread("Test_image.png")

bird_model = Gaussian_func(bird_mean, bird_cov)
sky_model = Gaussian_func(sky_mean, sky_cov)
cloud_model = Gaussian_func(cloud_mean, cloud_cov)

prediction_bird = np.sqrt(np.square(np.apply_along_axis(bird_model, 2, test_img)).sum(axis=2))
prediction_sky = np.sqrt(np.square(np.apply_along_axis(sky_model, 2, test_img)).sum(axis=2))
prediction_cloud = np.sqrt(np.square(np.apply_along_axis(cloud_model, 2, test_img)).sum(axis=2))


def Assign_label(b,s,c):
    b_norm = np.linalg.norm(b)
    s_norm = np.linalg.norm(s)
    c_norm = np.linalg.norm(c)
    
    if b_norm > s_norm and b_norm > c_norm:
        return 0
    if s_norm > b_norm and s_norm > c_norm:
        return 1
    return 2

def temp(a, b, c):
    print(a, b, c)
    return 0


bir = ((prediction_bird < prediction_cloud) < prediction_sky).astype(np.uint8) * 1
sk = ((prediction_sky > prediction_bird) > prediction_bird).astype(np.uint8) * 50
clou = ((prediction_cloud > prediction_bird) < prediction_sky).astype(np.uint8) * 200



test_label = bir+sk+clou
test_label = test_label.astype(np.uint8)


#cv2.imshow('test label',test_label)
cv2.imshow('test label',cv2.applyColorMap(test_label,cv2.COLORMAP_HSV))
#cv2.setMouseCallback("image",click_event,test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Image_test.png',test_label)

print(clou)


