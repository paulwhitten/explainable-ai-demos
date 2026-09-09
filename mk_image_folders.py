#!/usr/bin/env python

import os
import argparse

"""
This script creates image folders in the passed path 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trains transforms for a set')
    parser.add_argument('-f', '--folder', 
                        help='The folder to create image subfolders')
    args = parser.parse_args()

    folders = [
        "ellipse",
        "circle",
        "ellipse_circle",
        "skel",
        "fill",
        "skel_fill",
        "crossing",
        "endpoint",
        "lines",
        "chull",
        "thresh",
        "raw",
        "corner",
        "pca",
    ]

    for folder in folders:
        os.mkdir(os.path.join(args.folder, folder))

