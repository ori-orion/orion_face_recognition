#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 00:05:21 2022

@author: jianeng
"""

import rospy
#import sys
#sys.path.append('/home/jianeng/catkin_ws/src/actionserver_flib/src')
from actionserver_flib.actionserver_flib import AS_ClearDatabase #as as_flib


def main():
    rospy.init_node('as_cleardatabase_flib')
    as_Cleardatabase_obj = AS_ClearDatabase()    

    rospy.spin()


if __name__ == '__main__':
    main()