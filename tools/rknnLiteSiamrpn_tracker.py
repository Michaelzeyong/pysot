# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker

#rknn
# rknn
from rknnlite.api import RKNNLite
import cv2

class RKNNLiteSiamRPNTracker(object):
    def __init__(self,modelExemplarRknnName, modelInstanceRknnName,modelHeadRknnName):
        # super(RKNNSiamRPNTracker, self).__init__()

        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.initRKNN(modelExemplarRknnName, modelInstanceRknnName,modelHeadRknnName)



        # self.model = model
        # self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)

        return im_patch

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        # print('z_crop: ')
        # print(z_crop.size()) #torch.Size([1, 3, 127, 127])
        # self.model.template(z_crop)

        # read video and first img
        # cv2.namedWindow("video_name", cv2.WND_PROP_FULLSCREEN)
        # cap = cv2.VideoCapture(args.video_name)
        # ret, frameOri = cap.read()
        # frame = cv2.copyTo(frameOri,mask=None)
        # init_rect = cv2.selectROI("video_name", frame, False, False)
        
        # # print(init_rect)
        # x,y,w,h = init_rect
        # img = frame[y:y+h,x:x+w]
        # # cv2.imshow("video_name",img)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img,(127,127))
        # img = np.expand_dims(img, 0)

        back_T_in = z_crop.transpose((0,2,3,1))

        # Inference Exemplar input the object
        print('--> Running model Exemplar')
        # outputsExemplar = self.rknnExemplar.inference(inputs=[img], data_format=['nhwc'])
        self.outputsExemplar = self.rknnExemplar.inference(inputs=[back_T_in], data_format=['nhwc'])
        print(f"outputs len: {len(self.outputsExemplar)}")
        print(f"np shape  : {np.array(self.outputsExemplar).shape}")
        print('Inference Exemplar done')

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        # print('x_crop: ')
        # print(x_crop)
        # print(x_crop.size()) #torch.Size([1, 3, 287, 287])
        # outputs = self.model.track(x_crop)
        # print('outputs: ')
        # print(outputs)
        # score = self._convert_score(outputs['cls'])
        # pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        back_X_in = x_crop.transpose((0,2,3,1))
        # Inference InstanceName input the img
        # print('--> Running model Instance')
        # outputsInstance = self.rknnInstance.inference(inputs=[frame], data_format=['nhwc'])
        outputsInstance = self.rknnInstance.inference(inputs=[back_X_in], data_format=['nhwc'])
        # print(f"output len: {len(outputsInstance)}")
        # print(f"np shape  : {np.array(outputsInstance).shape}")
        # print('Inference Instance done')

        head_T_in = self.outputsExemplar[0].transpose((0,2,3,1))
        head_X_in = outputsInstance[0].transpose((0,2,3,1))

        

        # Inference Head
        # print('--> Running model Head')
        outputsHead = self.rknnHeadTrace.inference(inputs=[head_T_in,head_X_in])
        # print(outputsHead)
        # print(f"output len: {len(outputsHead)}")
        # print(f"np shape  : {outputsHead[0].shape}") #np shape  : (1, 10, 21, 21) cls
        # print(f"np shape  : {outputsHead[1].shape}") #np shape  : (1, 20, 21, 21) loc
        # print('Inference Head done')


        score = self._convert_score(torch.tensor(outputsHead[0]))
        pred_bbox = self._convert_bbox(torch.tensor(outputsHead[1]), self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }

    def initRKNN(self,modelExemplarRknnName, modelInstanceRknnName,modelHeadRknnName):

        # Exemplar init
        self.rknnExemplar = RKNNLite()

        # load Exemplar RKNN model
        print('--> Load Exemplar RKNN model')
        ret = self.rknnExemplar.load_rknn(modelExemplarRknnName)
        if ret != 0:
            print('Load Exemplar RKNN model failed')
            exit(ret)
        print('Load Exemplar RKNN model done!')

        # Init Exemplar runtime environment
        print('--> Init Exemplar runtime environment')
        ret = self.rknnExemplar.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print('Init Exemplar runtime environment failed!')
            exit(ret)
        print('Init Exemplar runtime environment done!')

        #----------------------------------------------------------------------------------------------

        # Instance init
        self.rknnInstance = RKNNLite()

        # load Instance RKNN model
        print('--> Load Instance RKNN model')
        ret = self.rknnInstance.load_rknn(modelInstanceRknnName)
        if ret != 0:
            print('Load Instance RKNN model failed')
            exit(ret)
        print('Load Instance RKNN model done!')
        
        # Init Instance runtime environment
        print('--> Init Instance runtime environment')
        ret = self.rknnInstance.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        if ret != 0:
            print('Init Instance runtime environment failed!')
            exit(ret)
        print('Init Instance runtime environment done!')

        #----------------------------------------------------------------------------------------------
        
        # Head init
        self.rknnHeadTrace = RKNNLite()

        # load Head RKNN model
        print('--> Load HeadTrace RKNN model')
        ret = self.rknnHeadTrace.load_rknn(modelHeadRknnName)
        if ret != 0:
            print('Load HeadTrace RKNN model failed')
            exit(ret)
        print('Load HeadTrace RKNN model done!')

        # Init Head runtime environment
        print('--> Init Head runtime environment')
        ret = self.rknnHeadTrace.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
        if ret != 0:
            print('Init Head runtime environment failed!')
            exit(ret)
        print('Init Head runtime environment Done')

    def releaseRKNN(self):
        self.rknnExemplar.release()
        self.rknnInstance.release()
        self.rknnHeadTrace.release()
