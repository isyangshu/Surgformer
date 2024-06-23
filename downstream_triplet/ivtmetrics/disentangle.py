#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An python implementation triplet component filtering .
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
(c) camma, icube, unistra
"""
#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np

#%%%%%%%%% COMPONENT FILTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Disentangle(object):
    """
    Class: filter a triplet prediction into the components (such as instrument i, verb v, target t, instrument-verb iv, instrument-target it, etc)    
    @args
    ----
        url: str. path to the dictionary map file of the dataset decomposition labels            
    @params
    ----------
    bank :   2D array
        holds the dictionary mapping of all components    
    @methods
    ----------
    extract(input, componet): 
        call filter a component labels from the inputs labels     
    """

    def __init__(self, url="maps.txt"):
        self.bank = self.map_file()
#        self.bank = np.genfromtxt(url, dtype=int, comments='#', delimiter=',', skip_header=0)
        
    def decompose(self, inputs, component):
        """ Extract the component labels from the triplets.
            @args:
                inputs: a 1D vector of dimension (n), where n = number of triplet classes;
                        with values int(0 or 1) for target labels and float[0, 1] for predicted labels.
                component: a string for the component to extract; 
                        (e.g.: i for instrument, v for verb, t for target, iv for instrument-verb pair, it for instrument-target pair and vt (unused) for verb-target pair)
            @return:
                output: int or float sparse encoding 1D vector of dimension (n), where n = number of component's classes.
        """
        txt2id = {'ivt':0, 'i':1, 'v':2, 't':3, 'iv':4, 'it':5, 'vt':6} 
        key    = txt2id[component]
        index  = sorted(np.unique(self.bank[:,key]))
        output = []
        for idx in index:
            same_class  = [i for i,x in enumerate(self.bank[:,key]) if x==idx]
            y           = np.max(np.array(inputs[same_class]))
            output.append(y)        
        return output
    
    def extract(self, inputs, component="i"):
        """
        Extract a component label from the triplet label
        @args
        ----
        inputs: 2D array,
            triplet labels, either predicted label or the groundtruth
        component: str,
            the symbol of the component to extract, choose from
            i: instrument
            v: verb
            t: target
            iv: instrument-verb
            it: instrument-target
            vt: verb-target (not useful)
        @return
        ------
        label: 2D array,
            filtered component's labels of the same shape and data type as the inputs
        """      
        if component == "ivt":
            return inputs
        else:
            component = [component]* len(inputs)
            return np.array(list(map(self.decompose, inputs, component)))

    def map_file(self):
        return np.array([ 
                       [ 0,  0,  2,  1,  2,  1],
                       [ 1,  0,  2,  0,  2,  0],
                       [ 2,  0,  2, 10,  2, 10],
                       [ 3,  0,  0,  3,  0,  3],
                       [ 4,  0,  0,  2,  0,  2],
                       [ 5,  0,  0,  4,  0,  4],
                       [ 6,  0,  0,  1,  0,  1],
                       [ 7,  0,  0,  0,  0,  0],
                       [ 8,  0,  0, 12,  0, 12],
                       [ 9,  0,  0,  8,  0,  8],
                       [10,  0,  0, 10,  0, 10],
                       [11,  0,  0, 11,  0, 11],
                       [12,  0,  0, 13,  0, 13],
                       [13,  0,  8,  0,  8,  0],
                       [14,  0,  1,  2,  1,  2],
                       [15,  0,  1,  4,  1,  4],
                       [16,  0,  1,  1,  1,  1],
                       [17,  0,  1,  0,  1,  0],
                       [18,  0,  1, 12,  1, 12],
                       [19,  0,  1,  8,  1,  8],
                       [20,  0,  1, 10,  1, 10],
                       [21,  0,  1, 11,  1, 11],
                       [22,  1,  3,  7, 13, 22],
                       [23,  1,  3,  5, 13, 20],
                       [24,  1,  3,  3, 13, 18],
                       [25,  1,  3,  2, 13, 17],
                       [26,  1,  3,  4, 13, 19],
                       [27,  1,  3,  1, 13, 16],
                       [28,  1,  3,  0, 13, 15],
                       [29,  1,  3,  8, 13, 23],
                       [30,  1,  3, 10, 13, 25],
                       [31,  1,  3, 11, 13, 26],
                       [32,  1,  2,  9, 12, 24],
                       [33,  1,  2,  3, 12, 18],
                       [34,  1,  2,  2, 12, 17],
                       [35,  1,  2,  1, 12, 16],
                       [36,  1,  2,  0, 12, 15],
                       [37,  1,  2, 10, 12, 25],
                       [38,  1,  0,  1, 10, 16],
                       [39,  1,  0,  8, 10, 23],
                       [40,  1,  0, 13, 10, 28],
                       [41,  1,  1,  2, 11, 17],
                       [42,  1,  1,  4, 11, 19],
                       [43,  1,  1,  0, 11, 15],
                       [44,  1,  1,  8, 11, 23],
                       [45,  1,  1, 10, 11, 25],
                       [46,  2,  3,  5, 23, 35],
                       [47,  2,  3,  3, 23, 33],
                       [48,  2,  3,  2, 23, 32],
                       [49,  2,  3,  4, 23, 34],
                       [50,  2,  3,  1, 23, 31],
                       [51,  2,  3,  0, 23, 30],
                       [52,  2,  3,  8, 23, 38],
                       [53,  2,  3, 10, 23, 40],
                       [54,  2,  5,  5, 25, 35],
                       [55,  2,  5, 11, 25, 41],
                       [56,  2,  2,  5, 22, 35],
                       [57,  2,  2,  3, 22, 33],
                       [58,  2,  2,  2, 22, 32],
                       [59,  2,  2,  1, 22, 31],
                       [60,  2,  2,  0, 22, 30],
                       [61,  2,  2, 10, 22, 40],
                       [62,  2,  2, 11, 22, 41],
                       [63,  2,  1,  0, 21, 30],
                       [64,  2,  1,  8, 21, 38],
                       [65,  3,  3, 10, 33, 55],
                       [66,  3,  5,  9, 35, 54],
                       [67,  3,  5,  5, 35, 50],
                       [68,  3,  5,  3, 35, 48],
                       [69,  3,  5,  2, 35, 47],
                       [70,  3,  5,  1, 35, 46],
                       [71,  3,  5,  8, 35, 53],
                       [72,  3,  5, 10, 35, 55],
                       [73,  3,  5, 11, 35, 56],
                       [74,  3,  2,  1, 32, 46],
                       [75,  3,  2,  0, 32, 45],
                       [76,  3,  2, 10, 32, 55],
                       [77,  4,  4,  5, 44, 65],
                       [78,  4,  4,  3, 44, 63],
                       [79,  4,  4,  2, 44, 62],
                       [80,  4,  4,  4, 44, 64],
                       [81,  4,  4,  1, 44, 61],
                       [82,  5,  6,  6, 56, 81],
                       [83,  5,  2,  2, 52, 77],
                       [84,  5,  2,  4, 52, 79],
                       [85,  5,  2,  1, 52, 76],
                       [86,  5,  2,  0, 52, 75],
                       [87,  5,  2, 10, 52, 85],
                       [88,  5,  7,  7, 57, 82],
                       [89,  5,  7,  4, 57, 79],
                       [90,  5,  7,  8, 57, 83],
                       [91,  5,  1,  0, 51, 75],
                       [92,  5,  1,  8, 51, 83],
                       [93,  5,  1, 10, 51, 85],
                       [94,  0,  9, 14,  9, 14],
                       [95,  1,  9, 14, 19, 29],
                       [96,  2,  9, 14, 29, 44],
                       [97,  3,  9, 14, 39, 59],
                       [98,  4,  9, 14, 49, 74],
                       [99,  5,  9, 14, 59, 89]
                       ])