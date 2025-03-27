from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import argparse

import cv2
import numpy as np
from glob import glob

import sys

sys.path.append(os.getcwd())

from pysot.core.config import cfg

from rknnLiteSiamrpn_tracker import RKNNLiteSiamRPNTracker

parser = argparse.ArgumentParser(description='tracking demo')

parser.add_argument('--config', default='./experiments/siamrpn_alex_dwxcorr/config.yaml', type=str, help='config file')
parser.add_argument('--video_name', default='./demo/bag.avi', type=str, help='config file')

args = parser.parse_args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break

    elif video_name.endswith('avi') or \
            video_name.endswith('mp4') or \
            video_name.endswith('mov'):
        cap = cv2.VideoCapture(video_name)

        # warmup
        for i in range(50):
            cap.read()

        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)

    # load_weight

    modelExemplarRknnName = './rknn/modelExemplar127.rknn'
    modelInstanceRknnName = './rknn/modelInstance287.rknn'
    modelHeadNameTraceRknnName = './rknn/modelHeadNameTrace.rknn'

    
    tracker = RKNNLiteSiamRPNTracker(modelExemplarRknnName, modelInstanceRknnName, modelHeadNameTraceRknnName)


    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    # calculate time 
    index = 0
    timestampStart = time.time();
    timestampEnd = time.time();
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
            print('Start tracking...')
        else:
            outputs = tracker.track(frame)
            # print(outputs)
            
            if index==0:
                timestampStart = time.time()
            if index ==10:
                timestampEnd = time.time()
                FPS = 10/(timestampEnd- timestampStart)
                print("FPS: %.2f" %FPS)
                index = 0
                timestampStart = time.time()
            

            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            key = cv2.waitKey(10)
            if key == ord('q') or key == 27:
                break

            index+=1
    
    tracker.releaseRKNN()


if __name__ == '__main__':
    main()
