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

    rospy.spin()


if __name__ == '__main__':
    main()