#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:52:07 2022

@author: jianeng
"""

import cv2
import rospy
import std_msgs.msg
#from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
import actionlib
import orion_face_recognition.msg

from cv_bridge import CvBridge, CvBridgeError

import time
import os
import glob


#!!!!!!!Note to change the system path
#import sys
#sys.path.append('/home/jianeng/Documents/My_code/DeepFace_tool')  # TODO - install this with pip
import flib.face_lib


# ActionServer CapFace
class AS_CapFace:
    
    def __init__(self, image_topic):
        self.flib_obj = flib.face_lib.Flib(self._get_current_image)
        
        # load tuning parameters to flib_obj
        self.flib_obj.set__wait_time(rospy.get_param('~wait_time'))
        self.flib_obj.set__max_face_num(rospy.get_param('~max_face_num'))
        self.flib_obj.set__num_representation(rospy.get_param('~num_representation'))
        self.flib_obj.set__model_name(rospy.get_param('~model_name'))
        self.flib_obj.set__distance_metric(rospy.get_param('~distance_metric'))
        self.flib_obj.set__detector_backend(rospy.get_param('~detector_backend'))
        self.flib_obj.set__normalization(rospy.get_param('~normalization'))
        
        
        # load path/dir parameters to flib_obj
        database_dir = rospy.get_param('~database_dir')
        if (database_dir[0]=='~'): # relative path
            database_dir=database_dir.replace('~','')
            database_dir=os.environ['HOME']+database_dir
            self.flib_obj.set__database_dir(database_dir)
        else: # absolute path
            self.flib_obj.set__database_dir(database_dir)
            
        
        flib_path = rospy.get_param('~flib_path')
        if (flib_path[0]=='~'): # relative path
            flib_path=flib_path.replace('~','')
            flib_path=os.environ['HOME']+flib_path
            self.flib_obj.set__script_path(flib_path)
        else: # absolute path
            self.flib_obj.set__script_path(flib_path)
        
        
        param_path = rospy.get_param('~param_path')
        if (param_path[0]=='~'): # relative path
            param_path=param_path.replace('~','')
            param_path=os.environ['HOME']+param_path
            self.flib_obj.set__param_path(param_path)
        else: # absolute path
            self.flib_obj.set__param_path(param_path)
        
        
        
        self.face_id=''
        
        
        self.img_topic = image_topic
        
        # Define bridge open cv -> RosImage
        self.bridge = CvBridge()
        
        self.as_Capface = actionlib.SimpleActionServer('as_Capface', orion_face_recognition.msg.ActionServer_CapFaceAction, execute_cb=self.Capface_cb, auto_start = False)
        
        self.If_subscribe=0
        
        #self.image_sub = rospy.Subscriber(image_topic, Image, callback)

        self.pub = rospy.Publisher('CapResult', Image, queue_size=10)

        self.as_Capface.start()
        
        rospy.loginfo('as_Capface action server initialized')

        rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image, self._image_callback)
        self._current_image = None

    def _image_callback(self, data):
        self._current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def _get_current_image(self):
        return self._current_image

    # as_Capface's callback function
    def Capface_cb(self,goal_msg):
        self.flib_obj.clearInterm_vars()
        self.flib_obj.clearBest_Face_Reps_ros()
        
        self.face_id=goal_msg.face_id
                
        frame_list_all=[]
        face_pixel_list_all = []
        
        
        #rate = rospy.Rate(100)
        timeout = rospy.get_time() + self.flib_obj.get__wait_time()
        
        
        
        rospy.loginfo('Start callback, time_exit: %s', str(timeout))
        
        idx_rep = 0; # Representation idx   
        while rospy.get_time() < timeout and idx_rep<self.flib_obj.get__num_representation():
            rospy.loginfo('Enter loop. time_now: %s',str(rospy.get_time()))
            #self.image_sub = rospy.Subscriber(self.img_topic, Image, self.img_sub_cb)
            
            while len(self.flib_obj.get__frame_list())<self.flib_obj.get__max_face_num() and rospy.get_time() < timeout:
                rospy.loginfo('Num frame: %s'+ str(len(self.flib_obj.get__frame_list())) )
                ros_image = rospy.wait_for_message(self.img_topic, Image)
                frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
                self.flib_obj.saveFace_ros(frame)
            
            
            frame_list_all.append(self.flib_obj.get__frame_list())
            face_pixel_list_all.append(self.flib_obj.get__face_pixel_list())
            
            ## Just debug
            # ff=self.flib_obj.get__frame_list()
            # for i in range(len(ff)):
            #     name='/home/jianeng/Pictures/FaceBaseDemo2/'+str(idx_rep)+'_'+str(i)+'.png'
            #     cv2.imwrite(name, ff[i])
                
                
            
            self.flib_obj.clearInterm_vars()
            
            idx_rep=idx_rep+1;

            rospy.sleep(1)
        
        rep_count=0
        for i in range(idx_rep):
            if (len(frame_list_all[i])==self.flib_obj.get__max_face_num()):
                rep_count=rep_count+1
                rospy.loginfo('Processing Rep: %s',str(i))
                self.flib_obj.replaceInterm_vars(frame_list=frame_list_all[i], face_pixel_list=face_pixel_list_all[i])
                Is_reg, best_idx, count_list = self.flib_obj.registerFace_ros()
                
                
                ## Just debug
                rospy.loginfo('Best idx: %s',str(best_idx))
                for nn in range(len(count_list)):
                    rospy.loginfo('Rep %s, Im %s, count: %s', str(idx_rep), str(nn), str(count_list[nn]))
        
        
        If_saved = self.flib_obj.saveBest_Face_Reps_ros(self.face_id)
        
        if (If_saved==True):
            rospy.loginfo('Image_saved')
            rospy.loginfo('rep num: %s', str(len(self.flib_obj.Best_Face_Reps)))
        else:
            rospy.loginfo('No image is saved')
        
        #self.as_Capface.result.name = 'Face'+str(self.face_id)

        if (If_saved==True):
            Im = self.flib_obj.Best_Face_Reps[0]#cv2.imread(matched_file_name)
            font = cv2.FONT_HERSHEY_COMPLEX
            Img_anno = cv2.putText(Im, 'Rep0', (0,50), font, 2, (255, 0, 0))
            Im_msg = self.bridge.cv2_to_imgmsg(Img_anno, "bgr8")
            self.pub.publish(Im_msg)

        
        _result = orion_face_recognition.msg.ActionServer_CapFaceResult()
        
        _result.name= self.face_id #  self.as_Capface.result.name
        _result.If_saved= If_saved
        
        self.as_Capface.set_succeeded(_result)

    # No longer need
    def img_sub_cb(self,ros_image):
        frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        self.flib_obj.saveFace_ros(frame)
        
        rospy.loginfo('In subscriber callback')
        
        if len(self.flib_obj.get__frame_list())>self.flib_obj.get__max_face_num():
            self.image_sub.unregister()
            rospy.loginfo('Unregister from the camera')
        
        

# ActionServer FindMatch
class AS_FindMatch:
    def __init__(self, image_topic):
        self.flib_obj = flib.face_lib.Flib()

        # load parameters to flib_obj
        self.flib_obj.set__wait_time(rospy.get_param('~wait_time'))
        self.flib_obj.set__max_face_num(rospy.get_param('~max_face_num'))
        self.flib_obj.set__num_representation(rospy.get_param('~num_representation'))
        self.flib_obj.set__model_name(rospy.get_param('~model_name'))
        self.flib_obj.set__distance_metric(rospy.get_param('~distance_metric'))
        self.flib_obj.set__detector_backend(rospy.get_param('~detector_backend'))
        self.flib_obj.set__normalization(rospy.get_param('~normalization'))
        
        
        # load path/dir parameters to flib_obj
        database_dir = rospy.get_param('~database_dir')
        if (database_dir[0]=='~'): # relative path
            database_dir=database_dir.replace('~','')
            database_dir=os.environ['HOME']+database_dir
            self.flib_obj.set__database_dir(database_dir)
        else: # absolute path
            self.flib_obj.set__database_dir(database_dir)
            
        
        flib_path = rospy.get_param('~flib_path')
        if (flib_path[0]=='~'): # relative path
            flib_path=flib_path.replace('~','')
            flib_path=os.environ['HOME']+flib_path
            self.flib_obj.set__script_path(flib_path)
        else: # absolute path
            self.flib_obj.set__script_path(flib_path)
        
        
        param_path = rospy.get_param('~param_path')
        if (param_path[0]=='~'): # relative path
            param_path=param_path.replace('~','')
            param_path=os.environ['HOME']+param_path
            self.flib_obj.set__param_path(param_path)
        else: # absolute path
            self.flib_obj.set__param_path(param_path)
        

        self.img_topic = image_topic
        
        # Define bridge open cv -> RosImage
        self.bridge = CvBridge()
        
        self.as_Findmatch = actionlib.SimpleActionServer('as_Findmatch', orion_face_recognition.msg.ActionServer_FindMatchAction, execute_cb=self.Findmatch_cb, auto_start = False)
        
        self.as_Findmatch.start()
        
        rospy.loginfo('as_Findmatch action server initialized')
        
        self.database_dir = self.flib_obj.get__database_dir()
        
        self.pub = rospy.Publisher('MatchResult', Image, queue_size=10)
        
        # Check how many img in the database
        current_dir = os.getcwd()
        os.chdir(self.database_dir)
        num_img = len(glob.glob('*.jpg*'))+len(glob.glob('*.png*'))
        rospy.loginfo('%s jpg or png files in the database',str(num_img))
        os.chdir(current_dir)
        
    # as_Findmatch's callback function
    def Findmatch_cb(self,goal_msg):
        self.flib_obj.clearInterm_vars()
        self.flib_obj.clearBest_Face_Reps_ros()
        
        timeout = rospy.get_time() + self.flib_obj.get__wait_time()
        
        rospy.loginfo('Start callback, time_exit: %s', str(timeout))
        
        while rospy.get_time() < timeout and len(self.flib_obj.get__frame_list())<=self.flib_obj.get__max_face_num():
            rospy.loginfo('Num frame: %s'+ str(len(self.flib_obj.get__frame_list())) )
            ros_image = rospy.wait_for_message(self.img_topic, Image)
            frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            self.flib_obj.saveFace_ros(frame)
        
        
        
        rospy.loginfo('Finish Capturing. Start finding match')
        
        If_find = self.flib_obj.FList_Find_ros()
        if (If_find == True):
            matched_file_name=self.flib_obj.best_match_dir
            match_score = self.flib_obj.best_match_score
            second_matched_file_name=self.flib_obj.second_best_match_dir
            second_matched_score = self.flib_obj.second_best_match_score
            
            
            # See if best match and second best match is the same person
            best_img_name=matched_file_name[len(self.database_dir):len(matched_file_name)]
            second_best_img_name=second_matched_file_name[len(self.database_dir):len(second_matched_file_name)]
            
            best_img_name_spilt = best_img_name.split('_')
            for i in range(len(best_img_name_spilt)):
                if (best_img_name_spilt[i]!='Face'):
                    face_id_best = best_img_name_spilt[i]
                    break
                
            second_best_img_name_split = second_best_img_name.split('_')
            for i in range(len(second_best_img_name_split)):
                if (second_best_img_name_split[i]!='Face'):
                    face_id_second_best = second_best_img_name_split[i]
                    break
            
            if face_id_best == face_id_second_best:
                match_score = match_score + second_matched_score
            
            
            # Match finish
            rospy.loginfo('Match Finish. File: %s, Score: %s', matched_file_name, str(match_score))
            
            _result = orion_face_recognition.msg.ActionServer_FindMatchResult()
            _result.file_name = matched_file_name
            _result.file_name_2ndbest = second_matched_file_name
            _result.best_match_score = match_score
            _result.second_best_match_score = second_matched_score
            _result.face_id = face_id_best
            _result.If_find=True
        
        else:
            rospy.loginfo('Match Finish. No match')
            
            _result = orion_face_recognition.msg.ActionServer_FindMatchResult()
            _result.file_name = ''
            _result.file_name_2ndbest = ''
            _result.best_match_score = 0
            _result.second_best_match_score = 0
            _result.face_id = ''
            _result.If_find=False
            
        if (_result.If_find==True):
            Im = cv2.imread(matched_file_name)
            font = cv2.FONT_HERSHEY_COMPLEX
            Img_anno = cv2.putText(Im, face_id_best, (0,50), font, 2, (255, 0, 0))
            Im_msg = self.bridge.cv2_to_imgmsg(Img_anno, "bgr8")
            self.pub.publish(Im_msg)
        
        # segment the matched_file_name to give the face_id
        # The naming of the registered face follows the following pattern
        # __database_dir + Face_ + face_name + _Rep_ + rep_count + .jpg
        # this means the first there is No word 'Face' appears, it gives the face_id (after spliting by '_')
        # txt_split = matched_file_name.split('_')
        # for i in range(len(txt_split)):
        #     if (txt_split[i].find('Face')==False):
        #         _result.face_id = txt_split[i]
        #         break
        
        self.as_Findmatch.set_succeeded(_result)





# ActionServer FindAttrs
class AS_FindAttrs:
    def __init__(self,image_topic):
        self.flib_obj = flib.face_lib.Flib()
        
        # load parameters to flib_obj
        self.flib_obj.set__wait_time(rospy.get_param('~wait_time'))
        self.flib_obj.set__max_face_num(rospy.get_param('~max_face_num'))
        self.flib_obj.set__num_representation(rospy.get_param('~num_representation'))
        self.flib_obj.set__model_name(rospy.get_param('~model_name'))
        self.flib_obj.set__distance_metric(rospy.get_param('~distance_metric'))
        self.flib_obj.set__detector_backend(rospy.get_param('~detector_backend'))
        self.flib_obj.set__normalization(rospy.get_param('~normalization'))
        
        # load path/dir parameters to flib_obj
        database_dir = rospy.get_param('~database_dir')
        if (database_dir[0]=='~'): # relative path
            database_dir=database_dir.replace('~','')
            database_dir=os.environ['HOME']+database_dir
            self.flib_obj.set__database_dir(database_dir)
        else: # absolute path
            self.flib_obj.set__database_dir(database_dir)
            
        
        flib_path = rospy.get_param('~flib_path')
        if (flib_path[0]=='~'): # relative path
            flib_path=flib_path.replace('~','')
            flib_path=os.environ['HOME']+flib_path
            self.flib_obj.set__script_path(flib_path)
        else: # absolute path
            self.flib_obj.set__script_path(flib_path)
        
        
        param_path = rospy.get_param('~param_path')
        if (param_path[0]=='~'): # relative path
            param_path=param_path.replace('~','')
            param_path=os.environ['HOME']+param_path
            self.flib_obj.set__param_path(param_path)
        else: # absolute path
            self.flib_obj.set__param_path(param_path)
        
        
        

        self.img_topic = image_topic
        
        # Define bridge open cv -> RosImage
        self.bridge = CvBridge()
        
        self.as_FindAttrs = actionlib.SimpleActionServer('as_Findattrs', orion_face_recognition.msg.ActionServer_FindAttrsAction, execute_cb=self.Findattrs_cb, auto_start = False)
        
        self.as_FindAttrs.start()

        self.failedface_counter=0

        
        rospy.loginfo('as_Findattrs action server initialized')
        
        
    def Findattrs_cb(self, goal_msg):
        self.flib_obj.clearInterm_vars()
        self.flib_obj.clearBest_Face_Reps_ros()
        
        
        face_id=goal_msg.face_id
        
        rospy.loginfo('Len face_id: %s', str(len(face_id)))
        
        #if you don't give face_id in goal_msg, it will use real-time camera to analyze facial attributes
        if (len(face_id)==0):
            timeout = rospy.get_time() + self.flib_obj.get__wait_time()
            
            rospy.loginfo('Start callback and capturing, time_exit: %s', str(timeout))
            
            while rospy.get_time() < timeout and len(self.flib_obj.get__frame_list())<=self.flib_obj.get__max_face_num():
                rospy.loginfo('Num frame: %s'+ str(len(self.flib_obj.get__frame_list())) )
                ros_image = rospy.wait_for_message(self.img_topic, Image)
                frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
                self.flib_obj.saveFace_ros(frame)

                ## timeout if no face found    
                if self.flib_obj.saveFace_ros(frame) == False:
                    self.failedface_counter += 1

                if self.failedface_counter == 15:
                    self.failedface_counter = 0
                    break

            
            
            
            rospy.loginfo('Finish Capturing. Start analyzing facial attributes')
            
            self.flib_obj.attributes_analy_ros()
            
        
        # if you give a face_id. The corresponding images in the database would be extracted and find attributes in it
        else:
            
            rospy.loginfo('No Capturing. Start loading imgs in database')
            
            current_dir = os.getcwd()
            
            database_dir = self.flib_obj.get__database_dir()
            
            os.chdir(database_dir)
            
            face_id_pattern='*'+face_id+'*'
            
            face_id_files = glob.glob(face_id_pattern)
            
            for i in face_id_files:
                frame = cv2.imread(i)
                self.flib_obj.saveFace_ros(frame)
            
            os.chdir(current_dir)
            
            rospy.loginfo('Finish loading. Start analyzing facial attributes')
            
            self.flib_obj.attributes_analy_ros()
            
            
            
        attr_list = self.flib_obj.attributes_list
        num_attrs = len(attr_list)
        
        if num_attrs==0:
            rospy.loginfo('No attributes detected')
        else:
            rospy.loginfo('Attrs:')
            for i in range(num_attrs):
                rospy.loginfo('%s',attr_list[i])
        
        
        _result = orion_face_recognition.msg.ActionServer_FindAttrsResult()
        
        
        for i in range(num_attrs):
             _result.attrs.append(attr_list[i])
        
        _result.num_attrs=num_attrs
        
        
        rospy.loginfo('Finish Attributes Analysis. Num Attrs found: %s', str(num_attrs))
        
        self.as_FindAttrs.set_succeeded(_result)
        
        
        

class AS_ClearDatabase:
    def __init__(self):
        self.flib_obj = flib.face_lib.Flib()

        # load parameters to flib_obj
        self.flib_obj.set__wait_time(rospy.get_param('~wait_time'))
        self.flib_obj.set__max_face_num(rospy.get_param('~max_face_num'))
        self.flib_obj.set__num_representation(rospy.get_param('~num_representation'))
        self.flib_obj.set__model_name(rospy.get_param('~model_name'))
        self.flib_obj.set__distance_metric(rospy.get_param('~distance_metric'))
        self.flib_obj.set__detector_backend(rospy.get_param('~detector_backend'))
        self.flib_obj.set__normalization(rospy.get_param('~normalization'))
        
        
        # load path/dir parameters to flib_obj
        database_dir = rospy.get_param('~database_dir')
        if (database_dir[0]=='~'): # relative path
            database_dir=database_dir.replace('~','')
            database_dir=os.environ['HOME']+database_dir
            self.flib_obj.set__database_dir(database_dir)
        else: # absolute path
            self.flib_obj.set__database_dir(database_dir)
            
        
        flib_path = rospy.get_param('~flib_path')
        if (flib_path[0]=='~'): # relative path
            flib_path=flib_path.replace('~','')
            flib_path=os.environ['HOME']+flib_path
            self.flib_obj.set__script_path(flib_path)
        else: # absolute path
            self.flib_obj.set__script_path(flib_path)
        
        
        param_path = rospy.get_param('~param_path')
        if (param_path[0]=='~'): # relative path
            param_path=param_path.replace('~','')
            param_path=os.environ['HOME']+param_path
            self.flib_obj.set__param_path(param_path)
        else: # absolute path
            self.flib_obj.set__param_path(param_path)
        
        
        
        self.database_dir = self.flib_obj.get__database_dir()
        
        self.as_ClearDatabase = actionlib.SimpleActionServer('as_Cleardatabase', orion_face_recognition.msg.ActionServer_ClearDatabaseAction, execute_cb=self.Cleardatabase_cb, auto_start = False)
        
        self.as_ClearDatabase.start()
        
        rospy.loginfo('as_Cleardatabase action server initialized')
        
        
    def Cleardatabase_cb(self,goal_msg):
        
        Is_success = False
        rospy.loginfo('Final Count')
        a=5
        for i in range(6):
            rospy.loginfo('%s',str(a))
            a=a-1
            rospy.sleep(1)
            
        
        current_dir = os.getcwd()
            
        os.chdir(self.database_dir)
        
        files = glob.glob('*')
        for f in files:
            os.remove(f)
        
        os.chdir(current_dir)
        Is_success = True
        
        _result = orion_face_recognition.msg.ActionServer_ClearDatabaseResult()
        _result.Is_success = Is_success
        
        rospy.loginfo('Delete all files in the database: %s', self.database_dir)
        
        self.as_ClearDatabase.set_succeeded(_result)







# main script test
if __name__ == '__main__':
    #FLib = flib.Flib()
    as_Capface_obj = AS_CapFace('/cv_camera/image_raw')
    
    rospy.init_node('AS_CapFace')
    
    rospy.spin()
    
    # # Ex_Find_Match_ros() OR Ex_attr_analysis_ros() Example code
    # timeout = time.time() + FLib.get__wait_time()
    
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()
    

    # while time.time() < timeout and len(FLib.get__frame_list())<=FLib.get__max_face_num():
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    
    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
        
    #     FLib.saveFace_ros(frame)
            
            
    
    # cap.release()
    
    # #b = FLib.FList_Find_ros()
    # #FLib.attributes_analy_ros()
