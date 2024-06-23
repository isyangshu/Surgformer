#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An python implementation detection AP metrics for surgical action triplet evaluation.
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
icube, unistra
"""
#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import sys
import warnings
from ivtmetrics.recognition import Recognition

#%%%%%%%%%% RECOGNITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Detection(Recognition):
    """
    Class: 
        input:
                video_gt: groundtruth json file with "frameid" as key
                video_pd: prediction json file with "frameid" as key
                json format:
                    frameid1: {
                                "recognition": List of class-wise probabilities in the 
                                    format:  [score1, score2, score3, ..., score100],
                                    example: [0.001, 0.025, 0.911, ...].
                                "detection" List of list of box detection for each triplet in the 
                                    format:  [[class1,score,x,y,w,h], [class3,score,x,y,w,h], [class56,score,x,y,w,h], ...],
                                    example: [[1,0.842,0.21,0.09,0.32,0.33], [3,0.992,0.21,0.09,0.32,0.33], [56,0.422,0.21,0.09,0.32,0.33], ....] 
                             }
                    frameid2: {...}
                    :
            output:
    
    @args
    ----
        num_class: 
            
    @params
    ----------
    bank :   2D array
        holds the dictionary mapping of all components    
    @methods
    -------
    extract(input, componet): 
        call filter a component labels from the inputs labels    
    impl: format: [{"triplet":tid, "instrument":[tool, 1.0, x,y,w,h], "target":[]}]
          add update(gt, pd) compute hits, ndet, npos, and record per frame/by/video
          add compute_AP('i/ivt') return AP for current run
          add compute_video_AP('i/ivt') return AP from video-wise averaging
          add compute_global_AP('i/ivt') return AP for all seen examples
          add reset_video()
    """
    def __init__(self, num_class=100, num_tool=6, threshold=0.5):
        super(Recognition, self).__init__()
        self.num_class      = num_class  
        self.num_tool       = num_tool                
        self.classwise_ap   = []
        self.classwise_rec  = []
        self.classwise_prec = []
        self.accumulator    = {}
        self.video_count    = 0
        self.end_call       = False
        self.threshold      = threshold
        self.reset()        
                
    def reset(self):
        self.video_count = 0
        self.video_end()  
        
    def reset_global(self):
        self.video_count = 0
        self.video_end()    
        
    def video_end(self):
        self.video_count += 1
        self.end_call = True
        self.accumulator[self.video_count] = {
                    "hits":  [[] for _ in range(self.num_class)],
                    "ndet":  [0  for _ in range(self.num_class)],
                    "npos":  [0  for _ in range(self.num_class)],                             
                    "hits_i":[[] for _ in range(self.num_tool)],
                    "ndet_i":[0  for _ in range(self.num_tool)] ,
                    "npos_i":[0  for _ in range(self.num_tool)] ,
                }
    
    def xywh2xyxy(self, bb):
        bb[2] += bb[0]
        bb[3] += bb[1]
        return bb    
    
    def iou(self, bb1, bb2):
        bb1 = self.xywh2xyxy(bb1)
        bb2 = self.xywh2xyxy(bb2)
        x1 = bb1[2] - bb1[0]
        y1 = bb1[3] - bb1[1]
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        x2 = bb2[2] - bb2[0]
        y2 = bb2[3] - bb2[1]
        if x2 < 0: x2 = 0
        if y2 < 0: y2 = 0
        xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
        yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
        if xiou < 0: xiou = 0
        if yiou < 0: yiou = 0
        if xiou * yiou <= 0:
            return 0
        else:
            return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)        
        
    def is_match(self, det_gt, det_pd, threshold):
        status = False    
        if det_gt[0] == det_pd[0]: # cond 1: correct identity        
            if self.iou(det_gt[-4:], det_pd[-4:]) >= threshold: # cond 2: sufficient iou
                status = True
        return status    
    
    def list2stack(self, x):
        if x == []: x = [[-1,-1,-1,-1,-1,-1]] # empty
        #x format for a single frame: list(list): each list = [tripletID, toolID, toolProbs, x, y, w, h] bbox is scaled (0..1)
        assert isinstance(x[0], list), "Each frame must be a list of lists, each list a prediction of triplet and object locations"
        x = np.stack(x, axis=0)
        x = x[x[:,2].argsort()[::-1]]
        return x    
    
    def sortstack(self, x):
        #x format for a single frame: list(list): each list = [tripletID, toolID, toolProbs, x, y, w, h] bbox is scaled (0..1)
        assert isinstance(x, np.ndarray), "Each frame must be an n-dim array with each row a unique prediction of triplet and object locations"
        x = x[x[:,2].argsort()[::-1]]
        return x
        
    def dict2stack(self, x):
        #x format for a single frame: list(dict): each dict = {"triplet":ID, "instrument": [ID, Probs, x, y, w, h]} bbox is scaled (0..1)
        assert isinstance(x, list), "Each frame must be a list of dictionaries"        
        y = []
        for d in x:
            assert isinstance(d, dict), "Each frame must be a list of dictionaries, each dictionary a prediction of triplet and object locations"
            p = [d['triplet']]
            p.extend(d["instrument"])
            y.append(p)
        return self.list2stack(y)    
    
    def update(self, targets, predictions, format="list"): 
        [self.update_frame(y, f, format) for y,f in zip(targets, predictions)]
#        print("First")
#        formats = [format]* len(targets)
#        map(self.update_frame, targets, predictions, formats)  
#        for item in range(len(targets)):
#            self.update_frame(targets[item], predictions[item], format)
        self.end_call = False 
    
    def update_frame(self, targets, predictions, format="list"):
#        print(self.end_call, "THIS GUY >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if format=="list":            
            detection_gt    = self.list2stack(targets)
            detection_pd    = self.list2stack(predictions)
        elif format=="dict":
            detection_gt    = self.dict2stack(targets)
            detection_pd    = self.dict2stack(predictions)
        else:
            sys.exit("unkown input format for update function. Must be a list or dict")
        if len(detection_pd) + len(detection_gt) == 0:
            return
        detection_gt_ivt = detection_gt.copy()
        detection_pd_ivt = detection_pd.copy()
        # for triplet
        for gt in detection_gt_ivt: 
            self.accumulator[self.video_count]["npos"][int(gt[0])] += 1
        for det_pd in detection_pd_ivt:
            self.accumulator[self.video_count]["ndet"][int(det_pd[0])] += 1
            matched = False
            for k, det_gt in enumerate(detection_gt_ivt):
                y = det_gt[0:] 
                f = det_pd[0:]
                if self.is_match(y, f, threshold=self.threshold):
                    detection_gt_ivt = np.delete(detection_gt_ivt, obj=k, axis=0)
                    matched = True
                    break
            if matched:
                self.accumulator[self.video_count]["hits"][int(det_pd[0])].append(1.0)
            else:
                self.accumulator[self.video_count]["hits"][int(det_pd[0])].append(0.0)
        # for instrument        
        detection_gt_i = detection_gt.copy()
        detection_pd_i = detection_pd.copy()
        for gt in detection_gt_i:
            self.accumulator[self.video_count]["npos_i"][int(gt[1])] += 1
        for det_pd in detection_pd_i:
            self.accumulator[self.video_count]["ndet_i"][int(det_pd[1])] += 1
            matched = False
            for k, det_gt in enumerate(detection_gt_i):                
                y = det_gt[1:] 
                f = det_pd[1:]
                if self.is_match(y, f, threshold=self.threshold):
                    detection_gt_i = np.delete(detection_gt_i, obj=k, axis=0)
                    matched = True
                    break
            if matched:
                self.accumulator[self.video_count]["hits_i"][int(det_pd[1])].append(1.0)
            else:
                self.accumulator[self.video_count]["hits_i"][int(det_pd[1])].append(0.0)  
        
                        
    def compute(self, component="ivt", video_id=None):
        classwise_ap    = []
        classwise_rec   = []
        classwise_prec  = []
        if video_id == None: 
            video_id = self.video_count-1 if self.end_call else self.video_count
        hit_str     = "hits" if component=="ivt" else "hits_i"
        pos_str     = "npos" if component=="ivt" else "npos_i"
        det_str     = "ndet" if component=="ivt" else "ndet_i"
        num_class   = self.num_class if component=="ivt" else self.num_tool       
        # decide on accumulator for framewise / video wise / current
        if video_id == -1:
            accumulator = {}
            accumulator[hit_str] = [sum([p[k]for p in [self.accumulator[f][hit_str] for f in self.accumulator] ],[]) for k in range(num_class)]            
            accumulator[pos_str] = list(np.sum(np.stack([self.accumulator[f][pos_str] for f in self.accumulator]), axis=0))       
            accumulator[det_str] = list(np.sum(np.stack([self.accumulator[f][det_str] for f in self.accumulator]), axis=0))
        else:
             accumulator = self.accumulator[video_id]        
        # compuatation
        for hits, npos, ndet in zip(accumulator[hit_str], accumulator[pos_str], accumulator[det_str]): # loop for num_class 
            if npos + ndet == 0: # no gt instance and no detection for the class
                classwise_ap.append(np.nan)
                classwise_rec.append(np.nan)
                classwise_prec.append(np.nan)
            elif npos>0 and len(hits)==0: # no detections but there are gt instances for the class
                classwise_ap.append(0.0)
                classwise_rec.append(0.0)
                classwise_prec.append(0.0)
            else:
                hits = np.cumsum(hits)
                ap   = 0.0
                rec  = hits / npos if npos else 0.0
                prec = hits / (np.array(range(len(hits)), dtype=np.float32) + 1.0)
                for i in range(11):
                    mask = rec >= (i / 10.0)
                    if np.sum(mask) > 0:
                        ap += np.max(prec[mask]) / 11.0
                classwise_ap.append(ap)
                classwise_rec.append(np.max(rec))
                classwise_prec.append(np.max(prec))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return (classwise_ap, np.nanmean(classwise_ap)), (classwise_rec, np.nanmean(classwise_rec)), (classwise_prec, np.nanmean(classwise_prec))
    
    def compute_video_AP(self, component="ivt"):
        classwise_ap    = []
        classwise_rec   = []
        classwise_prec  = []
        for j in range(self.video_count):
            video_id = j+1
            (ap, _), (rec, _), (prec, _) = self.compute(component=component, video_id=video_id)            
            classwise_ap.append(ap)
            classwise_rec.append(rec)
            classwise_prec.append(prec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            classwise_ap    = np.nanmean(np.stack(classwise_ap, axis=0), axis=0)
            classwise_rec   = np.nanmean(np.stack(classwise_rec, axis=0), axis=0)
            classwise_prec  = np.nanmean(np.stack(classwise_prec, axis=0), axis=0)        
            mAP             = np.nanmean(classwise_ap)
            mRec            = np.nanmean(classwise_rec)
            mPrec           = np.nanmean(classwise_prec) 
        return {"AP":classwise_ap, "mAP":mAP, "Rec":classwise_rec, "mRec":mRec, "Pre":classwise_prec, "mPre":mPrec}     
            
    def compute_AP(self, component="ivt"):
        a,r,p = self.compute(component=component, video_id=None)
        return {"AP":a[0], "mAP":a[1], "Rec":r[0], "mRec":r[1], "Pre":p[0], "mPre":p[1]}
        
    def compute_global_AP(self, component="ivt"):
        a,r,p =  self.compute(component=component, video_id=-1)
        return {"AP":a[0], "mAP":a[1], "Rec":r[0], "mRec":r[1], "Pre":p[0], "mPre":p[1]}
