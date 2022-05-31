#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 01:09:04 2022

@author: jianeng
"""
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
import face_lib as flib
import cv2
import time
import os


#import sys
#sys.path.append('/home/jianeng/Documents/My_code/DeepFace_tool')


model_name='Facenet'
metric_name='cosine'
num_face_all=1


database_dir = "/home/jianeng/Pictures/FaceBaseDemo/"


Flib_obj = flib.Flib()

#Flib_obj.attributes_realtime()


# # Capture Face with multiple representation
# Best_Face_Reps = flib.CapFaces_multi(wait_time = 10, max_face_num=10, num_representation=3) 


# rep_count=0;
# for i in range(len(Best_Face_Reps)):
#     if (isinstance(Best_Face_Reps[i],np.ndarray)): # if not empty
#         filename = database_dir + "Face_"+str(2)+"_Rep_"+str(rep_count)+".jpg"
#         cv2.imwrite(filename,Best_Face_Reps[i])
#         rep_count=rep_count+1



# #Find Best Match Face, Real-time
Flib_obj.FList_Find_realtime(db_path=database_dir, 
                                                                          wait_time=10, max_face_num=50,
                                                                          model_name=model_name, metric_name=metric_name)



#Best_match_dir, match_score, match_score_list 



    







# #Save One Face in the database
# for i in range(num_face_all):
#     filename = database_dir +"BestFace_" + str(i) + ".jpg"
    
#     msg = "Now capture: " + filename
#     print(msg)
#     time.sleep(5)
    
#     best_face = flib.CapFace(wait_time = 10, max_face_num=10) 
    
#     if (isinstance(best_face, np.ndarray)):
#         cv2.imwrite(filename,best_face)
    
    
# #Find Similar Face in the Database
# target_dir = "/home/jianeng/Pictures/FaceTargetDemo/"
# filename= target_dir+"My_target.jpg"
# best_face = flib.CapFace(wait_time = 10, max_face_num=10) 
# cv2.imwrite(filename,best_face)


# # Remove representations_vgg_face.pkl in database_dir
# if (model_name=='VGG-Face'):
#     remove_filename = database_dir + "representations_vgg_face.pkl"
# elif (model_name=='OpenFace'):
#     remove_filename = database_dir + "representations_openface.pkl"
# elif (model_name=='Facenet'):
#     remove_filename = database_dir + "representations_facenet.pkl"
# elif (model_name=='DeepFace'):
#     remove_filename = database_dir + "representations_deepface.pkl"
# elif (model_name=='DeepID'):
#     remove_filename = database_dir + "representations_deepid.pkl"

# if (os.path.exists(remove_filename)!=False):
#     os.remove(remove_filename)


# df = DeepFace.find(img_path = filename, db_path = database_dir,model_name=model_name)

# if (len(df)==0):
#     msg="No match!"
# else:
#     msg = "Best Match file: "+ df.loc[0].identity

# print(msg)




