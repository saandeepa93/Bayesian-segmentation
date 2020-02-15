import os
import numpy as np
import cv2

def insert_data(coord_lst, img,flag,prior_term,likelihood_term):
    print("inside insert data")
    sky_count_tmp = 0
    cloud_count_tmp = 0
    bird_count_tmp = 0
    for i in range(0,len(coord_lst),2):
        xmin, ymin = int(coord_lst[i][0]), int(coord_lst[i][1])
        xmax, ymax = int(coord_lst[i+1][0]), int(coord_lst[i+1][1])
        buff = img[xmin:xmax,ymin:ymax]
        uniq,cls_count = np.unique(buff,return_counts=True)
        if flag== 'sky':
            likelihood_term[uniq,0] += list(cls_count)
            sky_count_tmp += int(buff.shape[0])*int(buff.shape[1])
        elif flag == 'cloud':
            likelihood_term[uniq,1] += list(cls_count)
            cloud_count_tmp += int(buff.shape[0]*buff.shape[1])
        elif flag == 'bird':
            likelihood_term[uniq,2] += list(cls_count)
            bird_count_tmp += int(buff.shape[0]*buff.shape[1])
    return likelihood_term,sky_count_tmp,cloud_count_tmp,bird_count_tmp
    

def get_prior_and_likelihood(curr_dir,flag,normalizing_term,prior_term,likelihood_term):
    param =list()
    coord_lst = []
    count = 0
    sky_count = 0
    cloud_count = 0
    bird_count = 0
    
    for filename in os.listdir(curr_dir):
        if filename == 'img1.png_grayscale.png':
            param.append(coord_lst)
            img = cv2.imread(curr_dir+"\\"+filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(img.shape[0],img.shape[0]))
            count+=int(img.shape[1]*img.shape[0])
            uniq,tmp_count=np.unique(img,return_counts=True)
            tmp_count = np.reshape(tmp_count,(tmp_count.shape[0],1))
            normalizing_term[uniq] = tmp_count
            
            cv2.imshow("image",img)
            cv2.setMouseCallback("image",click_event,param)
            cv2.waitKey()
            cv2.destroyAllWindows  
                
            likelihood_term,sky_count, cloud_count, bird_count = insert_data(coord_lst,img,flag,prior_term,likelihood_term)
    return count,sky_count,cloud_count,bird_count,likelihood_term


def click_event(event, x, y, flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coord_dict = param[0]
        coord_dict.append([x,y])
        

        
'''
    * Prior: Probability that a pixel intensity appears throughout the image (255*1)
    * Likelihood: Probability that a pixel belongs to sky out of all the pixels (255*3)
    * Normalizing factor: total number of sky pixels/total number of pixels (3*1)
'''

def main():
    curr_dir = os.getcwd()+"\Grayscale"
    
    '''Initialise all the required variables'''
    count = 0
    sky_count = 0
    cloud_count = 0
    bird_count = 0
    normalizing_term = np.zeros(shape=(256,1))
    likelihood_term = np.zeros(shape=(256,3))
    prior_term = np.zeros(shape=(1,3))
    classes = ['sky','cloud','bird']
    print(np.amax(likelihood_term))
    
    #Get prior, likelihood and normalizing term
    for clas in classes:
        print("inside main loop")
        count,sky_count_tmp,cloud_count_tmp,bird_count_tmp,likelihood_term_tmp\
                = get_prior_and_likelihood(curr_dir,clas,normalizing_term,prior_term,likelihood_term)
        sky_count+=sky_count_tmp
        cloud_count+=cloud_count_tmp
        bird_count+=bird_count_tmp
        likelihood_term+=likelihood_term_tmp
    print(f"total count: {count}, sky count:{sky_count}, cloud count: {cloud_count}, bird count:{bird_count}\n")
    
    
    normalizing_term = normalizing_term / count
    likelihood_term[:,0] = likelihood_term[:,0] / sky_count
    likelihood_term[:,1] = likelihood_term[:,1] / cloud_count
    likelihood_term[:,2] = likelihood_term[:,2] / bird_count
    
    
    prior_term[:,0] = sky_count / count
    prior_term[:,1] = cloud_count / count
    prior_term[:,2] = bird_count / count

    
    '''
        calculate on test image
        Final probabilities
    '''
    test_img_path = os.getcwd()+"\Grayscale\Test_image.png"
    img_test = cv2.imread(test_img_path)
    img_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2GRAY)
    r,c = img_test.shape
    sky_prob = cloud_prob = bird_prob = np.zeros(shape=(r,c))
    
    '''Bayes function'''
    p_sky = np.vectorize(lambda val: (prior_term[0][0] * likelihood_term[val][0]) )
    p_cloud = np.vectorize(lambda val: (prior_term[0][1] * likelihood_term[val][1]))
    p_bird = np.vectorize(lambda val: (prior_term[0][2] * likelihood_term[val][2]))

    '''Calculate Bayesian probabilites'''
    sky_prob = p_sky(img_test)
    cloud_prob = p_cloud(img_test)
    bird_prob = p_bird(img_test)
    
    sky = ((sky_prob > cloud_prob) > bird_prob).astype(np.uint8)*1
    cloud = ((cloud_prob > sky_prob) > bird_prob).astype(np.uint8)*50
    bird = ((bird_prob > cloud_prob) > sky_prob).astype(np.uint8)*200
    
    test_label = bird+sky+cloud
    test_label = test_label.astype(np.uint8)
    
    
    #cv2.imshow('test label',test_label)
    cv2.imshow('test label',cv2.applyColorMap(test_label,cv2.COLORMAP_HSV))
    #cv2.setMouseCallback("image",click_event,test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Image_test.png',test_label)
    
if __name__=='__main__':
    main()
    
    
        