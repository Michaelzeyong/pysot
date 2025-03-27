from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
from xdrlib import ConversionError

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

import time
from rknnSiamrpn_tracker import RKNNSiamRPNTracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
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
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
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

def exportPt():
    print("--------------------Export *.pt--------------------")
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # EXEMPLAR_SIZE: 127
    # INSTANCE_SIZE: 287

    modelExemplarName = './rknn/modelExemplar127.pt'
    if not os.path.exists(modelExemplarName):   
        modelExemplar = torch.jit.trace(model.backbone,torch.Tensor(1,3,127,127))
        modelExemplar.save(modelExemplarName)
        print('save to %s' %modelExemplarName)
    else:
        print(f"Already have {modelExemplarName}")

    modelInstanceName = './rknn/modelInstance287.pt'
    if not os.path.exists(modelInstanceName): 
        modelInstance = torch.jit.trace(model.backbone,torch.Tensor(1,3,287,287))
        modelInstance.save(modelInstanceName)
        print('save to %s' %modelInstanceName)
    else:
        print(f"Already have {modelInstanceName}")

    # z_f shape: torch.Size([1, 256, 6, 6])
    # x_f shape: torch.Size([1, 256, 26, 26])

    # Use torch.jit.trace may have some problem
    # TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. 
    # We can't record the data flow of Python values, 
    # so this value will be treated as a constant in the future. 
    # This means that the trace might not generalize to other inputs!
    # out = F.conv2d(x, kernel, groups=batch*channel)

    modelHeadNameTrace = './rknn/modelHeadTrace.pt'
    if not os.path.exists(modelHeadNameTrace): 
        modelHead = torch.jit.trace(model.rpn_head, (torch.Tensor(1, 256, 6, 6), torch.Tensor(1, 256, 26, 26)))
        modelHead.save(modelHeadNameTrace)
        print('save to %s' %modelHeadNameTrace)
    else:
        print(f"Already have {modelHeadNameTrace}")
    
    # Use torch.jit.script
    modelHeadNameScript = './rknn/modelHeadscript.pt'
    if not os.path.exists(modelHeadNameScript):
        with torch.jit.optimized_execution(True): 
            modelHead = torch.jit.script(model.rpn_head, (torch.Tensor(1, 256, 6, 6), torch.Tensor(1, 256, 26, 26)))
            modelHead.save(modelHeadNameScript)
            print('save to %s' %modelHeadNameScript)
    else:
        print(f"Already have {modelHeadNameScript}")


    # print(model)
    print('Load Model Done')
    print("--------------------Export *.pt Done!--------------------")


def main():
    # load config
    cfg.merge_from_file(args.config)

    # export *.pt file
    exportPt()
    tracker = RKNNSiamRPNTracker()
    tracker.exportRKNN()
    tracker.initRKNN()

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
            print(outputs)
            
            if index==0:
                timestampStart = time.time()
            if index ==10:
                timestampEnd = time.time()
                FPS = 10/(timestampEnd- timestampStart)
                # print("FPS: %.2f" %FPS)
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
    

    

    

    # tracker = RKNNSiamRPNTracker()
    # tracker.init(frameOri, init_rect)


    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame,(287,287))
    # frame = np.expand_dims(frame, 0)

    


    

    # outputs = tracker.track(outputsHead)
    # print(outputs)
    # bbox = list(map(int, outputs['bbox']))
    # cv2.rectangle(frameOri, (bbox[0], bbox[1]),
    #                 (bbox[0]+bbox[2], bbox[1]+bbox[3]),
    #                 (0, 255, 0), 3)
    # cv2.imshow("video_name", frameOri)
    # key = cv2.waitKey(0)



    

    print('done')


if __name__ == '__main__':
    main()
