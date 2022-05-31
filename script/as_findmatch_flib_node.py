#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:06:55 2022

@author: jianeng
"""

import rospy
#import sys
#sys.path.append('/home/jianeng/catkin_ws/src/actionserver_flib/src')
from actionserver_flib.actionserver_flib import AS_FindMatch #as as_flib


def main():
    rospy.init_node('as_findmatch_flib')
    as_Findmatch_obj = AS_FindMatch(rospy.get_param('~img_topic'))
    
    # load parameters to flib_obj
    as_Findmatch_obj.flib_obj.set__wait_time(rospy.get_param('~wait_time'))
    as_Findmatch_obj.flib_obj.set__max_face_num(rospy.get_param('~max_face_num'))
    as_Findmatch_obj.flib_obj.set__num_representation(rospy.get_param('~num_representation'))
    as_Findmatch_obj.flib_obj.set__model_name(rospy.get_param('~model_name'))
    as_Findmatch_obj.flib_obj.set__distance_metric(rospy.get_param('~distance_metric'))
    as_Findmatch_obj.flib_obj.set__detector_backend(rospy.get_param('~detector_backend'))
    as_Findmatch_obj.flib_obj.set__normalization(rospy.get_param('~normalization'))
    as_Findmatch_obj.flib_obj.set__database_dir(rospy.get_param('~database_dir'))
    as_Findmatch_obj.flib_obj.set__script_path(rospy.get_param('~flib_path'))
    as_Findmatch_obj.flib_obj.set__param_path(rospy.get_param('~param_path'))
    
    
    

    rospy.spin()


if __name__ == '__main__':
    main()