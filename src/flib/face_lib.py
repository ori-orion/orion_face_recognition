
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 16:22:30 2022

Real-time Face Detector and Analyzier Usage Lib

@author: jianeng
"""

import cv2
import time
import numpy as np
from sys import exit
from deepface import DeepFace
from deepface.detectors import FaceDetector
from deepface.commons import functions
from deepface.commons import distance
import os
from tqdm import tqdm
import pickle
import pandas as pd


# force use of cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf_version = tf.__version__
#tf.device('/cpu:0') # force tf to use cpu
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    import keras
    from keras.preprocessing.image import load_img, save_img, img_to_array
    from keras.applications.imagenet_utils import preprocess_input
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image



# attributes related lib
#import pathlib
#import imutils
from pathlib import Path
#from fastai.vision.data import ImageList
# from fastai.vision.learner import create_cnn
import fastai
from fastai.vision import models
fastai_version = fastai.__version__
fastai_major_version = int(fastai_version.split(".")[0])
fastai_minor_version = int(fastai_version.split(".")[1])
if(fastai_major_version >=2 and fastai_minor_version >=6):
    from fastai.vision import *  # replaces the image modules import above
else:
    from fastai.vision.image import pil2tensor,Image
from fastai.vision import *  # replaces the image modules import above
#import csv







class Flib():

    def __init__(self, get_current_image = None):
        """
        Do change the parameters beforehand
        like: __databse_dir, __script_path
        """
        # Parameters
        self.__wait_time=100
        self.__max_face_num=30
        self.__num_representation=1
        self.__bboxsizelimit=100
        self.__model_name='VGG-Face'
        self.__distance_metric='cosine'
        self.__detector_backend='opencv'
        self.__normalization='base'
        
        self.__database_dir=''
        self.__script_path = ''
        self.__param_path = ''
        
        # Detectors
        with tf.device('/cpu:0'):
            self.model = DeepFace.build_model(self.__model_name)
        self.input_shape_x, self.input_shape_y = functions.find_input_shape(self.model)
        self.face_detector = FaceDetector.build_model(self.__detector_backend)
        
        
        
        # Intermediate variables
        self.__frame_list=[]
        self.__face_pixel_list=[]
        
        
        # Results
        self.Best_Face_Reps=[]
        self.best_match_dir=''
        self.best_match_score=0
        self.match_score_list={}
        
        self.second_best_match_dir=''
        self.second_best_match_score=0
        
        self.attributes_list=[]
        
        self.get_current_image = get_current_image

# Access private variables           
    def get__frame_list(self):
        return self.__frame_list
    
    def get__face_pixel_list(self):
        return self.__face_pixel_list
    
    def get__wait_time(self):
        return self.__wait_time
    
    def get__max_face_num(self):
        return self.__max_face_num
    
    def get__num_representation(self):
        return self.__num_representation
    
    def get__model_name(self):
        return self.__model_name
    
    def get__distance_metric(self):
        return self.__distance_metric
    
    def get__detector_backend(self):
        return self.__detector_backend
    
    def get__normalization(self):
        return self.__normalization
    
    def get__database_dir(self):
        return self.__database_dir

    def get__script_path(self):
        return self.__script_path
    
    def get__param_path(self):
        return self.__param_path
    
    
# Modify private variables
    def set__wait_time(self,value):
        self.__wait_time = value
    
    def set__max_face_num(self,value):
        self.__max_face_num = value
    
    def set__num_representation(self,value):
        self.__num_representation = value
    
    def set__model_name(self,value):
        self.__model_name = value
    
    def set__distance_metric(self,value):
        self.__distance_metric = value
        
    def set__detector_backend(self,value):
        self.__detector_backend = value
    
    def set__normalization(self,value):
        self.__normalization = value
    
    def set__database_dir(self,value):
        self.__database_dir = value
        # print('set database dir to: {}'.format(self.__database_dir))
    
    def set__script_path(self,value):
        self.__script_path= value

    def set__param_path(self,value):
        self.__param_path= value
    

# Clear variables
    def clearBest_Face_Reps_ros(self):
        self.Best_Face_Reps=[]
        
    def clearInterm_vars(self):
        self.__frame_list=[]
        self.__face_pixel_list=[]
        
    def replaceInterm_vars(self, frame_list, face_pixel_list):
        self.__frame_list=[]
        self.__face_pixel_list=[]
        self.__frame_list=frame_list
        self.__face_pixel_list=face_pixel_list
        
#Start: ROS related functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    def saveFace_ros(self, frame):
        """
        Input a cv2 frame, detect the face on the frame. If find face, save the frame to frame_list. Return If_saved
        """
        
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_face, img_region = FaceDetector.detect_face(face_detector=self.face_detector, detector_backend=self.__detector_backend, img=frame, align='True')
        
        If_saved=False
        if (isinstance(detected_face, np.ndarray)):
            det_color = self.ResizeImg(detected_face,target_size=(self.input_shape_y, self.input_shape_x))
                    
            img_pixels = image.img_to_array(det_color) #what this line doing? must?
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255 #normalize input in [0, 1]
                    
            self.__face_pixel_list.append(img_pixels)
            self.__frame_list.append(frame)
            
            If_saved= True
        
        return If_saved
        
    
    def registerFace_ros(self):
        """
        Go through the self.__face_pixel_list, find the best matched face. Register this face to Best_Face_Reps. Return If_registered
        """
        
        If_registered=False
        if (len(self.__frame_list) != 0):
            best_idx, count_list = self.FList_Compare(self.__face_pixel_list, model_name=self.__model_name,normalization=self.__normalization,distance_metric=self.__distance_metric,detector_backend=self.__detector_backend)
            self.Best_Face_Reps.append(self.__frame_list[best_idx])
            
            If_registered=True
        
        
        if (If_registered==False):
            best_idx=-1
            count_list=[]
        
        
        return If_registered,best_idx, count_list
    
    def saveBest_Face_Reps_ros(self, face_name):
        """
        Save Best_Face_Reps to the database
        Filename pattern:
        __database_dir + Face_ + face_name + _Rep_ + rep_count + .jpg
        """
        rep_count=0;
        for i in range(len(self.Best_Face_Reps)):
            if (isinstance(self.Best_Face_Reps[i],np.ndarray)): # if not empty
                filename = self.__database_dir + 'Face_' + face_name+"_Rep_"+str(rep_count)+".jpg"
                cv2.imwrite(filename,self.Best_Face_Reps[i])
                rep_count=rep_count+1
                
        if rep_count>0:
            If_saved=True
        else:
            If_saved=False
        
        return If_saved
                
    

    
    

    
    
    def FList_Find_ros(self):
        """
        Go through frame_list, find the most matched face in the database_dir. If success, return True
        """

        # if the frame list is empty, just return False
        if(len(self.__frame_list) == 0):
            return False

        self.best_match_dir=''
        self.best_match_score=0
        self.match_score_list={}
        self.second_best_match_dir=''
        self.second_best_match_score=0
        
        #model = DeepFace.build_model(model_name)
        #input_shape_x, input_shape_y = functions.find_input_shape(model)
        #face_detector = FaceDetector.build_model('opencv')
        
        #frame_list = []
        
        # Check If database folder exist
        if os.path.isdir(self.__database_dir) == False:
            raise ValueError("Passed db_path does not exist!")
            return False
        
        # used to find matched face in the database
        models_list = {}
        model_names_list=[]
        metric_names_list=[]
        models_list[self.__model_name]=self.model
        model_names_list.append(self.__model_name)
        metric_names_list.append(self.__distance_metric)
        
        
        # if representation .pkl file already exists, remove it
        remove_filename = "representations_%s.pkl" % (self.__model_name)
        remove_filename = remove_filename.replace("-", "_").lower()
        
        if (os.path.exists(self.__database_dir+remove_filename)!=False):
            os.remove(self.__database_dir+remove_filename)
            
            
        # Create representation .pkl file from scratch
        employees = []
    
        for r, d, f in os.walk(self.__database_dir): # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file.lower()) or ('.png' in file.lower()):
                    exact_path = r + file
                    employees.append(exact_path)
                    
        if len(employees) == 0:
            return False
            raise ValueError("There is no image in ", self.__database_dir," folder! Validate .jpg or .png files exist in this path.")
        
        #find representations for db images
        representation_list = []
        pbar = tqdm(range(0,len(employees)), desc='Finding representation_list', disable = True)
        
        #for employee in employees:
        for index in pbar:
            employee = employees[index]
    
            instance = []
            instance.append(employee)
    
            for j in model_names_list:
                custom_model = models_list[j]
                with tf.device('/cpu:0'):
                    representation = DeepFace.represent(img_path = employee
                                    , model_name = self.__model_name, model = self.model
                                    , enforce_detection = False, detector_backend = self.__detector_backend
                                    , align = True
                                    , normalization = self.__normalization)
    
                    instance.append(representation)
                
            # representation_list is a list of img_pixels for each file in the database
            representation_list.append(instance)
    
        f = open(self.__database_dir+remove_filename, "wb")
        pickle.dump(representation_list, f)
        f.close()
        
        #~~~~~~~~~~~~~~~~~~~~now, we got representations for facial database~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # create a vector to count each img in the database's match time
        
        
        for i in range(len(employees)):
            self.match_score_list[employees[i]]=0
        
        
        df = pd.DataFrame(representation_list, columns = ["identity", "%s_representation" % (self.__model_name)])
        df_base = df.copy()
        
        
        
        # # Capture Faces
        # timeout = time.time() + wait_time
        # cap = cv2.VideoCapture(0)
        # if not cap.isOpened():
        #     print("Cannot open camera")
        #     exit()
            
        # while time.time() < timeout and len(frame_list)<max_face_num:
        #     # Capture frame-by-frame
        #     ret, frame = cap.read()
            
        #     # if frame is read correctly ret is True
        #     if not ret:
        #         print("Can't receive frame (stream end?). Exiting ...")
        #         break
        #      # Our operations on the frame come here
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
        #     detected_face, img_region = FaceDetector.detect_face(face_detector=face_detector, detector_backend='opencv', img=frame, align='True')
                
        #     # Display the resulting frame, gray OR frame
        #     cv2.imshow('frame', frame)
                
                
        #     # If face detected, Save the frame in a list
        #     if (isinstance(detected_face, np.ndarray)):
        #         frame_list.append(frame)
        
        # # When everything done, release the capture
        # cap.release()        
        
        for idx_frame in range(len(self.__frame_list)):
            
            
            resp_obj = []
            
            global_pbar = tqdm(range(0,1), desc='Analyzing', disable = True)
            
            for j in global_pbar:
                for j in model_names_list:
                    custom_model = models_list[j]
                    
                    # Represent detected frame
                    img_pixels = self.FLib_preprocess_face(img = self.__frame_list[idx_frame], target_size=(self.input_shape_y, self.input_shape_x))
                    with tf.device('/cpu:0'): #force cpu
                        target_representation = self.model.predict(img_pixels)[0].tolist()
            
                    for k in metric_names_list:
                        distances = []
                        for index, instance in df.iterrows():
                            source_representation = instance["%s_representation" % (j)]

                            if k == 'cosine':
                                distance_Value = distance.findCosineDistance(source_representation, target_representation)
                            elif k == 'euclidean':
                                distance_Value = distance.findEuclideanDistance(source_representation, target_representation)
                            elif k == 'euclidean_l2':
                                distance_Value = distance.findEuclideanDistance(distance.l2_normalize(source_representation), distance.l2_normalize(target_representation))
    
    
                            distances.append(distance_Value)
                            
                        df["%s_%s" % (j, k)] = distances
    
                        if self.__model_name != 'Ensemble':
                            threshold = distance.findThreshold(j, k)
                            df = df.drop(columns = ["%s_representation" % (j)])
                            df = df[df["%s_%s" % (j, k)] <= threshold]
    
                            df = df.sort_values(by = ["%s_%s" % (j, k)], ascending=True).reset_index(drop=True)
    
                            resp_obj.append(df)
                            df = df_base.copy() #restore df for the next iteration
                       
            if (len(resp_obj)!=0):
                if (len(resp_obj[0])!=0):
                    self.match_score_list[resp_obj[0].loc[0].identity]=self.match_score_list[resp_obj[0].loc[0].identity]+1
    
        
        match_score_list_sorted = sorted(self.match_score_list.items(), key=lambda x:x[1],reverse=True)
        
        self.best_match_dir = match_score_list_sorted[0][0]
        self.best_match_score = match_score_list_sorted[0][1]/len(self.__frame_list)
        if (len(match_score_list_sorted)>=2):
            self.second_best_match_dir = match_score_list_sorted[1][0]
            self.second_best_match_score = match_score_list_sorted[1][1]/len(self.__frame_list)
        
        
        # max_count=0
        # for i in self.match_score_list:
        #     if self.match_score_list[i]>max_count:
        #         max_count=self.match_score_list[i]
        #         self.best_match_dir = i
        
        
        # self.best_match_score = self.match_score_list[self.best_match_dir]/len(self.__frame_list)
        
        print("Task Complete")
    
        return True



    def attributes_analy_ros(self):
        self.attributes_list=[]
        history=[]
        
        path=self.__script_path#Path(os.getcwd())
        #fastai.device = torch.device('cpu') #force cpu
        #with torch.device("cpu"):
        learn=load_learner(path,"ff_stage-1-256-rn50.pkl",device='cpu')
        face_cascade = cv2.CascadeClassifier(path+"haarcascade_frontalface_default.xml")
        
        for i in range(len(self.__frame_list)):
            gray = cv2.cvtColor(self.__frame_list[i], cv2.COLOR_BGR2GRAY)
            face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            ## Looping through each face
            for coords in face_coord:
                
                ## Finding co-ordinates of face
                X, Y, w, h = coords
    
                ## Finding frame size
                H, W, _ = self.__frame_list[i].shape

                if w + h <self.__bboxsizelimit:
                    continue
    
                ## Computing larger face co-ordinates
                X_1, X_2 = (max(0, X - int(w * 0.35)), min(X + int(1.35 * w), W))
                Y_1, Y_2 = (max(0, Y - int(0.35 * h)), min(Y + int(1.35 * h), H))
    
                ## Cropping face and changing BGR To RGB
                img_cp = self.__frame_list[i][Y_1:Y_2, X_1:X_2].copy()
                img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
                
                
                #fastai.device = torch.device('cpu')# Prediction of facial featues
                prediction = str(
                    learn.predict(Image(pil2tensor(img_cp1, np.float32).div_(255)))[0]
                ).split(";")
                label = (
                    " ".join(prediction)
                    if "Male" in prediction
                    else "Female " + " ".join(prediction)
                )
                label = (
                    " ".join(prediction)
                    if "No_Beard" in prediction
                    else "Beard " + " ".join(prediction)
                )
    
    
                ## Drawing facial boundaries
                cv2.rectangle(
                    img=self.__frame_list[i],
                    pt1=(X, Y),
                    pt2=(X + w, Y + h),
                    color=(255, 0, 0),
                    thickness=1,
                )
    
                ## Drawing facial attributes identified
                label_list = label.split(" ")
    
                history+=label_list
                [self.attributes_list.append(x) for x in history if x not in self.attributes_list and x != "Blurry" 
                and x != "5_o_Clock_Shadow" and x != "Mouth_Slightly_Open" and x != "Smiling"]
                
                if "Beard" in self.attributes_list and "No_Beard" in self.attributes_list:
                    self.attributes_list.remove("No_Beard")
    
    
        print(self.attributes_list)
        
        
        


# End: ROS related functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  


# Start: Example ros callback~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def Ex_CapFace_ros(self):
        
        """
        NEVER call this function. Replace all self with Flib_obj
        """
        # Flib_obj = Flib.Flib() # in the action server class, you need to instantiate the object
        
        frame_list_all=[]
        face_pixel_list_all = []
        
        
        timeout = time.time() + self.get__wait_time()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
        idx_rep = 0; # Representation idx    
        while time.time() < timeout and idx_rep<self.get__num_representation():
            
            while time.time() < timeout and len(self.get__frame_list())<=self.get__max_face_num():
                # Capture frame-by-frame
                ret, frame = cap.read()
            
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                
                self.saveFace_ros(frame)
                
            
            frame_list_all.append(self.get__frame_list())
            face_pixel_list_all.append(self.get__face_pixel_list())
            
            self.clearInterm_vars()
            
            idx_rep=idx_rep+1;

            time.sleep(1)
        
        cap.release()
        
        for i in range(idx_rep):
            self.replaceInterm_vars(frame_list=frame_list_all[i], face_pixel_list=face_pixel_list_all[i])
            self.registerFace_ros()
        
        self.saveBest_Face_Reps_ros('me')
        

    def Ex_Find_Match_ros(self):
        """
        NEVER call this function. Replace all self with Flib_obj
        """
        # Flib_obj = Flib.Flib() # in the action server class, you need to instantiate the object
        
        
        
        timeout = time.time() + self.get__wait_time()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
            
            while time.time() < timeout and len(self.get__frame_list())<=self.get__max_face_num():
                # Capture frame-by-frame
                ret, frame = cap.read()
            
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                
                self.saveFace_ros(frame)
                
        
        cap.release()
        
        self.FList_Find_ros()


    def Ex_attr_analysis_ros(self):
        """
        NEVER call this function. Replace all self with Flib_obj
        """
        # Flib_obj = Flib.Flib() # in the action server class, you need to instantiate the object
        
        
        timeout = time.time() + self.get__wait_time()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
            
            while time.time() < timeout and len(self.get__frame_list())<=self.get__max_face_num():
                # Capture frame-by-frame
                ret, frame = cap.read()
            
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                
                self.saveFace_ros(frame)
                
        
        cap.release()

        self.attributes_analy_ros()


# End: Example ros callback~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def print_info(self):
        print("""Fuck You!""")
        
        
        
    def WebCam_on(self,wait_time):
        """
        Turn on the WebCam for wait_time second
    
        """
        
        timeout = time.time() + wait_time
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while time.time() < timeout:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    
        
    
    
    
    # Capture One Face's multiple representations
    def CapFaces_multi(self,wait_time, max_face_num=30, num_representation=3, model_name = 'VGG-Face', distance_metric = 'cosine'):
        """
        Turn on WebCam, Capture faces on the same person
        with multiple most frequently appeared faces as different representation of the same face . And the face list
        """
        
        self.Best_Face_Reps=[]
        with tf.device('/cpu:0'):
            model = DeepFace.build_model(model_name)
        input_shape_x, input_shape_y = functions.find_input_shape(model)
        face_detector = FaceDetector.build_model('opencv')
        face_list=[] # grayscale face list
        face_pixel_list=[] # Used to Compare faces, part of functions.preprocess_face()
        img_list=[] # whole grayscale image list
        frame_list=[] # whole colorful image list
        frame_list_all=[] # Total frame list to save each "frame_list"
        face_pixel_list_all = [] # # Total face pixel list saved in "face_pixel_list_all"
        
        
        # Create num_representation's empty element of frame_list_all
        for i in range(num_representation):
            frame_list_all.append([])
            face_pixel_list_all.append([])
        
        
        # Capture Faces
        timeout = time.time() + wait_time
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
        idx_rep = 0; # Representation idx    
        while time.time() < timeout and idx_rep<num_representation:
            
            while time.time() < timeout and len(face_list)<=max_face_num:
                # Capture frame-by-frame
                ret, frame = cap.read()
            
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                detected_face, img_region = FaceDetector.detect_face(face_detector=face_detector, detector_backend='opencv', img=frame, align='True')
                
                # Display the resulting frame, gray OR frame
                cv2.imshow('frame', frame)
                
                
                # Add detected face to the list
                if (isinstance(detected_face, np.ndarray)):
                    det_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                    #det_gray = ResizeImg(det_gray,target_size=(input_shape_y, input_shape_x))
                    det_color = self.ResizeImg(detected_face,target_size=(input_shape_y, input_shape_x))
                    
                    img_pixels = image.img_to_array(det_color) #what this line doing? must?
                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                    img_pixels /= 255 #normalize input in [0, 1]
                    
                    face_pixel_list.append(img_pixels)
                    face_list.append(det_gray)
                    img_list.append(gray)
                    frame_list.append(frame)
                    #cv2.imshow('frame', detected_face)
            
            frame_list_all[idx_rep].extend(frame_list)
            face_pixel_list_all[idx_rep].extend(face_pixel_list)
            
            face_pixel_list.clear()
            face_list.clear()
            img_list.clear()
            frame_list.clear()
            
            idx_rep=idx_rep+1;
            
            time.sleep(1)
            
            
            if cv2.waitKey(1) == ord('q'):
                break
    
        # When everything done, release the capture
        cap.release()
        #cv2.destroyAllWindows()
        
        #return frame_list_all
        
        
        for idx in range(idx_rep):
            if (len(frame_list_all[idx]) != 0):
                best_idx = self.FList_Compare(face_pixel_list_all[idx], model_name=model_name, distance_metric=distance_metric)
                self.Best_Face_Reps.append(frame_list_all[idx][best_idx])
        
        
        print("Task Complete")
        
        #return Best_Face_Reps
                
            
    
    
    # Capture One Representation of One Face
    def CapFace(self, wait_time, max_face_num=5, model_name = 'VGG-Face'):
        """
        Turn on WebCam, Capture the most frequently appeared face. And return the face
        """
        self.Best_Face_Reps=[]
        with tf.device('/cpu:0'):
            model = DeepFace.build_model(model_name)
        input_shape_x, input_shape_y = functions.find_input_shape(model)
        face_detector = FaceDetector.build_model('opencv')
        face_list=[] # grayscale face list
        face_pixel_list=[] # Used to Compare faces, part of functions.preprocess_face()
        img_list=[] # whole grayscale image list
        frame_list=[] # whole colorful image list
        
        
      
        
        timeout = time.time() + wait_time
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while time.time() < timeout and len(face_list)<=max_face_num:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            detected_face, img_region = FaceDetector.detect_face(face_detector=face_detector, detector_backend='opencv', img=frame, align='True')
            
            
            
            # Display the resulting frame, gray OR frame
            cv2.imshow('frame', frame)
            
            # Add detected face to the list
            if (isinstance(detected_face, np.ndarray)):
                det_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                #det_gray = ResizeImg(det_gray,target_size=(input_shape_y, input_shape_x))
                det_color = self.ResizeImg(detected_face,target_size=(input_shape_y, input_shape_x))
                
                img_pixels = image.img_to_array(det_color) #what this line doing? must?
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255 #normalize input in [0, 1]
                
                face_pixel_list.append(img_pixels)
                face_list.append(det_gray)
                img_list.append(gray)
                frame_list.append(frame)
            #cv2.imshow('frame', detected_face)
            
            if cv2.waitKey(1) == ord('q'):
                break
            
        
        
        # When everything done, release the capture
        cap.release()
        #cv2.destroyAllWindows()
        
        #return face_list
        
        # Find the most frequently appeared face and return it
        if (len(face_list) != 0):
            best_idx = self.FList_Compare(face_pixel_list)
            #return face_list[best_idx] #In case you want to save face
            #return img_list[best_idx] #In case you want to save img
            #return frame_list[best_idx] #In case you want to save colorful frame
            self.Best_Face_Reps.append(frame_list[best_idx])
        #else: # If no face exist   
            #return face_list
        
        
    def ResizeImg(self, gray_frame, target_size=(224, 224),grayscale=False):
        """
        Resize the detected face to be compatible with DeepFace 
        
        """
        if gray_frame.shape[0] > 0 and gray_frame.shape[1] > 0:
            factor_0 = target_size[0] / gray_frame.shape[0]
            factor_1 = target_size[1] / gray_frame.shape[1]
            factor = min(factor_0, factor_1)
       
            dsize = (int(gray_frame.shape[1] * factor), int(gray_frame.shape[0] * factor))
            gray_frame = cv2.resize(gray_frame, dsize)
       
       		# Then pad the other side to the target size by adding black pixels
            diff_0 = target_size[0] - gray_frame.shape[0]
            diff_1 = target_size[1] - gray_frame.shape[1]
            if grayscale == False:
       			# Put the base image in the middle of the padded image
                gray_frame = np.pad(gray_frame, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
            else:
                gray_frame = np.pad(gray_frame, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')
       
       	#------------------------------------------
       
       	#double check: if target image is not still the same size with target.
        if gray_frame.shape[0:2] != target_size:
            gray_frame = cv2.resize(gray_frame, target_size)
           
        return gray_frame
        
        
    def FList_Compare(self, face_list, model_name = 'VGG-Face',normalization = 'base', distance_metric = 'cosine', detector_backend='opencv'):
        with tf.device('/cpu:0'):
            model = DeepFace.build_model(model_name)
        metric_name = distance_metric
        
        embedding_face_list = []
        
        size_list = len(face_list)
        
        # If just one or two faces detected, just use the first one (idx=0)
        if (size_list<=2):
            return 0
        
        else:
            # Similar_Count_list counts the number of faces that are similar to the current idx one
            Similar_Count_list=[]
            for i in range(size_list):
                # Statistics of each face's similar faces count in the face_list
                Similar_Count_list.append(1)
                
                # represents facial images as vectors
                #normalize_face=functions.normalize_input(img = face_list[i], normalization = normalization)
                #embedding_face_list.append(model.predict(normalize_face)[0].tolist())
    
    
            for i in range(size_list):
                normalize_face=functions.normalize_input(img = face_list[i], normalization = normalization)
                current_rep = model.predict(normalize_face)[0].tolist()
                
                for j in range(size_list):
                    if (j !=i):
                        normalize_face=functions.normalize_input(img = face_list[j], normalization = normalization)
                        compare_rep=model.predict(normalize_face)[0].tolist()
                        
                        if metric_name == 'cosine':
                            distance_ = distance.findCosineDistance(current_rep, compare_rep)
                        elif metric_name == 'euclidean':
                            distance_ = distance.findEuclideanDistance(current_rep, compare_rep)
                        elif metric_name == 'euclidean_l2':
                            distance_ = distance.findEuclideanDistance(distance.l2_normalize(current_rep), distance.l2_normalize(compare_rep))
    
                        distance_ = np.float64(distance_)
    
                        threshold = distance.findThreshold(model_name, metric_name)
                        #threshold = threshold/2
    
                        if distance_ <= threshold:
                            identified = True
                            Similar_Count_list[i]=Similar_Count_list[i]+1
                        else:
                            identified = False
    
                        """
                            resp_obj = {
                                "verified": identified
                                , "distance": distance
                                , "threshold": threshold
                                , "model": model_name
                                , "detector_backend": detector_backend
                                , "similarity_metric": distance_metric
                            }
                        """
                        
            # Find the best index of the face that has the most match
            # If we have multiple faces have most matches, select the smallest index
            best_idx = Similar_Count_list.index(max(Similar_Count_list))
            
            return best_idx, Similar_Count_list
    
    
    
    def FLib_preprocess_face(self, img, target_size=(224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv', return_region = False, align = True):
        
        img, region = functions.detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection, align = align)
        
        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("Detected face shape is ", img.shape,". Not an image")
        
        #post-processing
        if grayscale == True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        #---------------------------------------------------
        #resize image to expected shape
    
        # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image
    
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = target_size[0] / img.shape[0]
            factor_1 = target_size[1] / img.shape[1]
            factor = min(factor_0, factor_1)
    
            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)
    
            # Then pad the other side to the target size by adding black pixels
            diff_0 = target_size[0] - img.shape[0]
            diff_1 = target_size[1] - img.shape[1]
            if grayscale == False:
                # Put the base image in the middle of the padded image
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
            else:
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')
    
    	#------------------------------------------
    
        #double check: if target image is not still the same size with target.
        if img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)
    
        #---------------------------------------------------
    
        #normalizing the image pixels
    
        img_pixels = image.img_to_array(img) #what this line doing? must?
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 #normalize input in [0, 1]
    
        #---------------------------------------------------
    
        if return_region == True:
            return img_pixels, region
        else:
            return img_pixels
    
    
    
    
    def FList_Find_realtime(self, db_path,wait_time=10,model_name='VGG-Face', metric_name='cosine', max_face_num=30, prog_bar = True, normalization = 'base', enforce_detection=False, align=True, detector_backend = 'opencv'):
        """
        Turn on Camera. Capture several faces, then compare these faces with the database. Find out the most likely match
        """
        
        
        self.best_match_dir=''
        self.best_match_score=0
        self.match_score_list={}
        with tf.device('/cpu:0'):
            model = DeepFace.build_model(model_name)
        input_shape_x, input_shape_y = functions.find_input_shape(model)
        face_detector = FaceDetector.build_model('opencv')
        
        frame_list = []
        
        # Check If database folder exist
        if os.path.isdir(db_path) == False:
            raise ValueError("Passed db_path does not exist!")
            return None
        
        # used to find matched face in the database
        models = {}
        model_names_list=[]
        metric_names_list=[]
        models[model_name]=model
        model_names_list.append(model_name)
        metric_names_list.append(metric_name)
        
        
        # if representation .pkl file already exists, remove it
        remove_filename = "representations_%s.pkl" % (model_name)
        remove_filename = remove_filename.replace("-", "_").lower()
        
        if (os.path.exists(db_path+remove_filename)!=False):
            os.remove(db_path+remove_filename)
            
            
        # Create representation .pkl file from scratch
        employees = []
    
        for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file.lower()) or ('.png' in file.lower()):
                    exact_path = r + file
                    employees.append(exact_path)
                    
        if len(employees) == 0:
            raise ValueError("There is no image in ", db_path," folder! Validate .jpg or .png files exist in this path.")
        
        #find representations for db images
        representation_list = []
        pbar = tqdm(range(0,len(employees)), desc='Finding representation_list', disable = prog_bar)
        
        
        #for employee in employees:
        to_remove_in_employees=[]
        for index in pbar:
            employee = employees[index]
    
            instance = []
            instance.append(employee)
    
            for j in model_names_list:
                custom_model = models[j]
                with tf.device('/cpu:0'):
                    representation = DeepFace.represent(img_path = employee
                                    , model_name = model_name, model = custom_model
                                    , enforce_detection = enforce_detection, detector_backend = detector_backend
                                    , align = align
                                    , normalization = normalization)
                #if(isinstance(representation,np.ndarray)):
                instance.append(representation)
                # representation_list is a list of img_pixels for each file in the database
                representation_list.append(instance)
                #else:
                #    to_remove_in_employees.append(employee)
        
        # return to_remove_in_employees, employees
        # for i in range(len(to_remove_in_employees)):
        #     employees.remove(to_remove_in_employees[i])
                
            
    
        f = open(db_path+remove_filename, "wb")
        pickle.dump(representation_list, f)
        f.close()
        
        #return employees
        #~~~~~~~~~~~~~~~~~~~~now, we got representations for facial database~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # create a vector to count each img in the database's match time
        
        for i in range(len(employees)):
            self.match_score_list[employees[i]]=0
        
        
        df = pd.DataFrame(representation_list, columns = ["identity", "%s_representation" % (model_name)])
        df_base = df.copy()
        
        
        
        # Capture Faces
        timeout = time.time() + wait_time
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
            
        while time.time() < timeout and len(frame_list)<max_face_num:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
             # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            detected_face, img_region = FaceDetector.detect_face(face_detector=face_detector, detector_backend='opencv', img=frame, align='True')
                
            # Display the resulting frame, gray OR frame
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
                
            # If face detected, Save the frame in a list
            if (isinstance(detected_face, np.ndarray)):
                frame_list.append(frame)
        
        # When everything done, release the capture
        cap.release()        
        
        
        for idx_frame in range(len(frame_list)):
            
            
            resp_obj = []
            
            global_pbar = tqdm(range(0,1), desc='Analyzing', disable = prog_bar)
            
            for j in global_pbar:
                for j in model_names_list:
                    custom_model = models[j]
                    
                    # Represent detected frame
                    img_pixels = self.FLib_preprocess_face(img = frame_list[idx_frame], target_size=(input_shape_y, input_shape_x))
                    target_representation = model.predict(img_pixels)[0].tolist()
            
                    for k in metric_names_list:
                        distances = []
                        for index, instance in df.iterrows():
                            source_representation = instance["%s_representation" % (j)]
    
                            if k == 'cosine':
                                distance_Value = distance.findCosineDistance(source_representation, target_representation)
                            elif k == 'euclidean':
                                distance_Value = distance.findEuclideanDistance(source_representation, target_representation)
                            elif k == 'euclidean_l2':
                                distance_Value = distance.findEuclideanDistance(distance.l2_normalize(source_representation), distance.l2_normalize(target_representation))
    
                            distances.append(distance_Value)
                            
                        df["%s_%s" % (j, k)] = distances
    
                        if model_name != 'Ensemble':
                            threshold = distance.findThreshold(j, k)
                            df = df.drop(columns = ["%s_representation" % (j)])
                            df = df[df["%s_%s" % (j, k)] <= threshold]
    
                            df = df.sort_values(by = ["%s_%s" % (j, k)], ascending=True).reset_index(drop=True)
    
                            resp_obj.append(df)
                            df = df_base.copy() #restore df for the next iteration
                       
            if (len(resp_obj)!=0):
                if (len(resp_obj[0])!=0):
                    self.match_score_list[resp_obj[0].loc[0].identity]=self.match_score_list[resp_obj[0].loc[0].identity]+1
    
        
    
        match_score_list_sorted = sorted(self.match_score_list.items(), key=lambda x:x[1],reverse=True)
        
        self.best_match_dir = match_score_list_sorted[0][0]
        self.best_match_score = match_score_list_sorted[0][1]/max_face_num
        self.second_best_match_dir = match_score_list_sorted[1][0]
        self.second_best_match_score = match_score_list_sorted[1][1]/max_face_num
    
        # max_count=0
        # for i in self.match_score_list:
        #     if self.match_score_list[i]>max_count:
        #         max_count=self.match_score_list[i]
        #         self.best_match_dir = i
        
        # self.best_match_score = self.match_score_list[self.best_match_dir]/max_face_num
        
        print("Task Complete")
        
        #return best_match_dir, match_score_list[best_match_dir]/max_face_num, match_score_list
    
    
    #def find_BestMatch()
    
    
    def attributes_oneframe(self, img_path='', wait_time=10,max_face_num=5, model_name='VGG-Face'):
        """
        if img_path is not empty, check the attributes of the input frame
        else turn on the camera and check the attributes of the detected frame
        """
        
        self.attributes_list=[]
        with tf.device('/cpu:0'):
            model = DeepFace.build_model(model_name)
        input_shape_x, input_shape_y = functions.find_input_shape(model)
        face_detector = FaceDetector.build_model('opencv')
        history=[]
        path=Path(os.getcwd())
        learn=load_learner(path,"ff_stage-1-256-rn50.pkl")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        if img_path!="":
            OneFace = cv2.imread(img_path)
            
            #cv2.imshow('frame', OneFace)
        
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            timeout = time.time() + wait_time
            frame_list=[]
            face_pixel_list=[]
            while time.time() < timeout and len(frame_list)<=max_face_num:
                ret, frame = cap.read()
            
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                detected_face, img_region = FaceDetector.detect_face(face_detector=face_detector, detector_backend='opencv', img=frame, align='True')
            
            
            
                # Display the resulting frame, gray OR frame
                #cv2.imshow('frame', frame)
            
                # Add detected face to the list
                if (isinstance(detected_face, np.ndarray)):
                    det_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                    #det_gray = ResizeImg(det_gray,target_size=(input_shape_y, input_shape_x))
                    det_color = self.ResizeImg(detected_face,target_size=(input_shape_y, input_shape_x))
                    
                    img_pixels = image.img_to_array(det_color) #what this line doing? must?
                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                    img_pixels /= 255 #normalize input in [0, 1]
                    
                    face_pixel_list.append(img_pixels)
                    #face_list.append(det_gray)
                    #img_list.append(gray)
                    frame_list.append(frame)
                    #cv2.imshow('frame', detected_face)
            
                if cv2.waitKey(1) == ord('q'):
                    break
            
        
        
            # When everything done, release the capture
            cap.release()
            #cv2.destroyAllWindows()
        
        
            # Find the most frequently appeared face and return it
            if (len(frame_list) != 0):
                best_idx = self.FList_Compare(face_pixel_list)
                #return face_list[best_idx] #In case you want to save face
                #return img_list[best_idx] #In case you want to save img
                #return frame_list[best_idx] #In case you want to save colorful frame
                OneFace = frame_list[best_idx]

        face_coord = face_cascade.detectMultiScale(OneFace, 1.1, 5, minSize=(30, 30))
        ## Looping through each face
        for coords in face_coord:
            
            ## Finding co-ordinates of face
            X, Y, w, h = coords

            ## Finding frame size
            H, W, _ = OneFace.shape

            if w + h <self.__bboxsizelimit:
                continue

            ## Computing larger face co-ordinates
            X_1, X_2 = (max(0, X - int(w * 0.35)), min(X + int(1.35 * w), W))
            Y_1, Y_2 = (max(0, Y - int(0.35 * h)), min(Y + int(1.35 * h), H))

            ## Cropping face and changing BGR To RGB
            img_cp = OneFace[Y_1:Y_2, X_1:X_2].copy()
            img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)

            ## Prediction of facial featues
            prediction = str(
                learn.predict(Image(pil2tensor(img_cp1, np.float32).div_(255)))[0]
            ).split(";")
            label = (
                " ".join(prediction)
                if "Male" in prediction
                else "Female " + " ".join(prediction)
            )
            label = (
                " ".join(prediction)
                if "No_Beard" in prediction
                else "Beard " + " ".join(prediction)
            )

            ## Drawing facial boundaries
            cv2.rectangle(
                img=OneFace,
                pt1=(X, Y),
                pt2=(X + w, Y + h),
                color=(255, 0, 0),
                thickness=1,
            )
            
            ## Drawing facial attributes identified
            label_list = label.split(" ")

            history+=label_list
            [self.attributes_list.append(x) for x in history if x not in self.attributes_list and x != "Blurry" 
            and x != "5_o_Clock_Shadow" and x != "Mouth_Slightly_Open" and x != "Smiling"]
            
            if "Beard" in self.attributes_list and "No_Beard" in self.attributes_list:
                self.attributes_list.remove("No_Beard")
            
            cv2.imshow("frame", OneFace)
            
            if cv2.waitKey(1) == ord('q'):
                    break

        
        
    def attributes_realtime(self, wait_time=10,max_face_num=200):
        self.attributes_list=[]
        history=[]
        
        path=Path(os.getcwd())
        learn=load_learner(path,"ff_stage-1-256-rn50.pkl")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # cap = cv2.VideoCapture(0)
        # if not cap.isOpened():
        #     print("Cannot open camera")
        #     exit()
            
        timeout = time.time() + wait_time
        
        count_read=0
        while time.time() < timeout and count_read<=max_face_num:
            # ret , frame = cap.read()

            # if not ret:
            #     print("Can't receive frame (stream end?). Exiting ...")
            #     break

            frame = self.get_current_image()

            count_read=count_read+1
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            ## Looping through each face
            for coords in face_coord:
                
                ## Finding co-ordinates of face
                X, Y, w, h = coords

                if w + h <self.__bboxsizelimit:
                    continue
    
                ## Finding frame size
                H, W, _ = frame.shape
    
                ## Computing larger face co-ordinates
                X_1, X_2 = (max(0, X - int(w * 0.35)), min(X + int(1.35 * w), W))
                Y_1, Y_2 = (max(0, Y - int(0.35 * h)), min(Y + int(1.35 * h), H))
    
                ## Cropping face and changing BGR To RGB
                img_cp = frame[Y_1:Y_2, X_1:X_2].copy()
                img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
    
                ## Prediction of facial featues
                prediction = str(
                    learn.predict(Image(pil2tensor(img_cp1, np.float32).div_(255)))[0]
                ).split(";")
                label = (
                    " ".join(prediction)
                    if "Male" in prediction
                    else "Female " + " ".join(prediction)
                )
                label = (
                    " ".join(prediction)
                    if "No_Beard" in prediction
                    else "Beard " + " ".join(prediction)
                )

                ## Drawing facial boundaries
                cv2.rectangle(
                    img=frame,
                    pt1=(X, Y),
                    pt2=(X + w, Y + h),
                    color=(255, 0, 0),
                    thickness=1,
                )
    
                ## Drawing facial attributes identified
                label_list = label.split(" ")
    
                history+=label_list
                [self.attributes_list.append(x) for x in history if x not in self.attributes_list and x != "Blurry" 
                and x != "5_o_Clock_Shadow" and x != "Mouth_Slightly_Open" and x != "Smiling"]
                
                if "Beard" in self.attributes_list and "No_Beard" in self.attributes_list:
                    self.attributes_list.remove("No_Beard")
    
    
                print(self.attributes_list)
            
            
            
            
# main script test
if __name__ == '__main__':
    FLib = Flib()
    database_dir = "/home/jianeng/Pictures/FaceBaseDemo/"
    #FLib.FList_Find_realtime(db_path=database_dir)
    #FLib.CapFace(wait_time=10)
    #FLib.attributes_oneframe(img_path='/home/jianeng/Pictures/profile.png')
    #FLib.attributes_realtime()
    
    
    # # Ex_CapFace_ros()  Example code
    # frame_list_all=[]
    # face_pixel_list_all = []
    
    
    # timeout = time.time() + FLib.get__wait_time()
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()
    
    # idx_rep = 0; # Representation idx    
    # while time.time() < timeout and idx_rep<FLib.get__num_representation():
        
    #     while time.time() < timeout and len(FLib.get__frame_list())<=FLib.get__max_face_num():
    #         # Capture frame-by-frame
    #         ret, frame = cap.read()
        
    #         # if frame is read correctly ret is True
    #         if not ret:
    #             print("Can't receive frame (stream end?). Exiting ...")
    #             break
            
    #         FLib.saveFace_ros(frame)
            
        
    #     frame_list_all.append(FLib.get__frame_list())
    #     face_pixel_list_all.append(FLib.get__face_pixel_list())
        
    #     FLib.clearInterm_vars()
        
    #     idx_rep=idx_rep+1;

    #     time.sleep(1)
    
    # cap.release()
    
    # for i in range(idx_rep):
    #     FLib.replaceInterm_vars(frame_list=frame_list_all[i], face_pixel_list=face_pixel_list_all[i])
    #     FLib.registerFace_ros()
    
    # FLib.saveBest_Face_Reps_ros('me')

    


    # # Ex_Find_Match_ros() OR Ex_attr_analysis_ros() Example code
    timeout = time.time() + FLib.get__wait_time()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    

    while time.time() < timeout and len(FLib.get__frame_list())<=FLib.get__max_face_num():
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        FLib.saveFace_ros(frame)
            
            
    
    cap.release()
    
    b = FLib.FList_Find_ros()
    #FLib.attributes_analy_ros()