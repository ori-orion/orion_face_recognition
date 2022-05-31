#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:15:35 2022

@author: jianeng
"""

import rospy
#import sys
#sys.path.append('/home/jianeng/catkin_ws/src/actionserver_flib/src')
from actionserver_flib.actionserver_flib import AS_CapFace #as as_flib



def main():
    rospy.init_node('as_capface_flib')
    as_Capface_obj = AS_CapFace(rospy.get_param('~img_topic'))
    
    
    
    
    rospy.spin()


if __name__ == '__main__':
    main()