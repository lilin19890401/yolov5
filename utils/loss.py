# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel
import cv2
import numpy as np


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# 自己加的代码
###########################################################################################################
###########################################################################################################
def merge_imgs(imgs, row_col_num):
    """
        Merges all input images as an image with specified merge format.

        :param imgs : img list
        :param row_col_num : number of rows and columns displayed
        :return img : merges img
        """

    length = len(imgs)
    row, col = row_col_num

    assert row > 0 or col > 0, 'row and col cannot be negative at same time!'
    color = random_color(rgb=True).astype(np.float64)

    for img in imgs:
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color)

    if row_col_num[1] < 0 or length < row:
        merge_imgs = np.hstack(imgs)
    elif row_col_num[0] < 0 or length < col:
        merge_imgs = np.vstack(imgs)
    else:
        assert row * col >= length, 'Imgs overboundary, not enough windows to display all imgs!'

        fill_img_list = [np.zeros(imgs[0].shape, dtype=np.uint8)] * (row * col - length)
        imgs.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(row):
            start = col * i
            end = col * (i + 1)
            merge_col = np.hstack(imgs[start: end])
            merge_imgs_col.append(merge_col)

        merge_imgs = np.vstack(merge_imgs_col)

    return merge_imgs


def show_img(imgs, window_names=None, wait_time_ms=0, is_merge=False, row_col_num=(1, -1)):
    """
        Displays an image or a list of images in specified windows or self-initiated windows.
        You can also control display wait time by parameter 'wait_time_ms'.
        Additionally, this function provides an optional parameter 'is_merge' to
        decide whether to display all imgs in a particular window 'merge'.
        Besides, parameter 'row_col_num' supports user specified merge format.
        Notice, specified format must be greater than or equal to imgs number.

        :param imgs: numpy.ndarray or list.
        :param window_names: specified or None, if None, function will create different windows as '1', '2'.
        :param wait_time_ms: display wait time.
        :param is_merge: whether to merge all images.
        :param row_col_num: merge format. default is (1, -1), image will line up to show.
                            example=(2, 5), images will display in two rows and five columns.
        """
    if not isinstance(imgs, list):
        imgs = [imgs]

    if window_names is None:
        window_names = list(range(len(imgs)))
    else:
        if not isinstance(window_names, list):
            window_names = [window_names]
        assert len(imgs) == len(window_names), 'window names does not match images!'

    if is_merge:
        merge_imgs1 = merge_imgs(imgs, row_col_num)

        cv2.namedWindow('merge', 0)
        cv2.imshow('merge', merge_imgs1)
    else:
        for img, win_name in zip(imgs, window_names):
            if img is None:
                continue
            win_name = str(win_name)
            cv2.namedWindow(win_name, 0)
            cv2.imshow(win_name, img)

    cv2.waitKey(wait_time_ms)


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def show_bbox(image, bboxs_list, color=None,
              thickness=1, font_scale=0.3, wait_time_ms=0, names=None,
              is_show=True, is_without_mask=False):
    """
    Visualize bbox in object detection by drawing rectangle.

    :param image: numpy.ndarray.
    :param bboxs_list: list: [pts_xyxy, prob, id]: label or prediction.
    :param color: tuple.
    :param thickness: int.
    :param fontScale: float.
    :param wait_time_ms: int
    :param names: string: window name
    :param is_show: bool: whether to display during middle process
    :return: numpy.ndarray
    """
    assert image is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_copy = image.copy()
    for bbox in bboxs_list:
        if len(bbox) == 5:
            txt = '{:.3f}'.format(bbox[4])
        elif len(bbox) == 6:
            txt = 'p={:.3f},id={:.3f}'.format(bbox[4], bbox[5])
        bbox_f = np.array(bbox[:4], np.int32)
        if color is None:
            colors = random_color(rgb=True).astype(np.float64)
        else:
            colors = color

        if not is_without_mask:
            image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors,
                                       thickness)
        else:
            mask = np.zeros_like(image_copy, np.uint8)
            mask1 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, -1)
            mask = np.zeros_like(image_copy, np.uint8)
            mask2 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, thickness)
            mask2 = cv2.addWeighted(mask1, 0.5, mask2, 8, 0.0)
            image_copy = cv2.addWeighted(image_copy, 1.0, mask2, 0.6, 0.0)
        if len(bbox) == 5 or len(bbox) == 6:
            cv2.putText(image_copy, txt, (bbox_f[0], bbox_f[1] - 2),
                        font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    if is_show:
        show_img(image_copy, names, wait_time_ms)
    return image_copy


def xywhToxyxy(bbox):
    y = np.zeros_like(bbox)
    y[:, 0] = bbox[:, 0] - bbox[:, 2] / 2  # top left x
    y[:, 1] = bbox[:, 1] - bbox[:, 3] / 2  # top left y
    y[:, 2] = bbox[:, 0] + bbox[:, 2] / 2  # bottom right x
    y[:, 3] = bbox[:, 1] + bbox[:, 3] / 2  # bottom right y
    return y


def vis_bbox(imgs, targets):
    tar = targets.cpu().detach().numpy()
    data = imgs * 255
    data = data.permute(0, 2, 3, 1).cpu().detach().numpy()
    h, w = data.shape[1], data.shape[1]
    gain = np.ones(6)
    gain[2:6] = np.array([w, h, w, h])
    tar = (tar * gain)
    for i in range(imgs.shape[0]):
        img = data[i].astype(np.uint8)
        img = img[..., ::-1]
        tar1 = tar[tar[:, 0] == i][:, 2:]
        y = xywhToxyxy(tar1)
        show_bbox(img, y)


def vis_match(imgs, targets, tcls, tboxs, indices, anchors, pred, ttars):
    tar = targets.cpu().detach().numpy()
    data = imgs * 255
    data = data.permute(0, 2, 3, 1).cpu().detach().numpy()
    h, w = data.shape[1], data.shape[2]
    gain = np.ones(6)
    gain[2:6] = np.array([w, h, w, h])
    tar = (tar * gain)

    strdie = [8, 16, 32]
    # 对每张图片进行可视化
    for j in range(imgs.shape[0]):
        img = data[j].astype(np.uint8)[..., ::-1]
        tar1 = tar[tar[:, 0] == j][:, 2:]
        y1 = xywhToxyxy(tar1)
        # img = VisualHelper.show_bbox(img1.copy(), y1, color=(255, 255, 255), is_show=False, thickness=2)
        # 对每个预测尺度进行单独可视化
        vis_imgs = []
        for i in range(3):  # i=0检测小物体，i=1检测中等尺度物体，i=2检测大物体
            s = strdie[i]
            # anchor尺度
            gain1 = np.array(pred[i].shape)[[3, 2, 3, 2]]
            b, a, gx, gy = indices[i]
            b1 = b.cpu().detach().numpy()
            gx1 = gx.cpu().detach().numpy()
            gy1 = gy.cpu().detach().numpy()
            anchor = anchors[i].cpu().detach().numpy()
            ttar = ttars[i].cpu().detach().numpy()

            # 找出对应图片对应分支的信息
            indx = b1 == j
            gx1 = gx1[indx]
            gy1 = gy1[indx]
            anchor = anchor[indx]
            ttar = ttar[indx]

            # 还原到原图尺度进行可视化
            ttar /= gain1
            ttar *= np.array([w, h, w, h], np.float32)
            y = xywhToxyxy(ttar)
            # label 可视化
            img1 = show_bbox(img.copy(), y, color=(0, 0, 255), is_show=False)

            # anchor 需要考虑偏移，在任何一层，每个bbox最多3*3=9个anchor进行匹配
            anchor *= s
            anchor_bbox = np.stack([gy1, gx1], axis=1)
            k = np.array(pred[i].shape, np.float)[[3, 2]]
            anchor_bbox = anchor_bbox / k
            anchor_bbox *= np.array([w, h], np.float32)
            anchor_bbox = np.concatenate([anchor_bbox, anchor], axis=1)
            anchor_bbox1 = xywhToxyxy(anchor_bbox)
            # 正样本anchor可视化
            img1 = show_bbox(img1, anchor_bbox1, color=(0, 255, 255), is_show=False)
            vis_imgs.append(img1)
        show_img(vis_imgs, is_merge=True)

###########################################################################################################
###########################################################################################################




# 核心内容
def compute_loss(p, targets, model, imgs=None):  # predictions, targets, model
    # 可视化target
    if imgs != None:
        vis_bbox(imgs, targets)

    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors, ttar = build_targets(p, targets, model)  # targets: classid, box ratio, image, anchor, grid indices, anchors,  box
    # 可视化anchor匹配关系
    if imgs != None:
        vis_match(imgs, targets, tcls, tbox, indices, anchors, p, ttar)

    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))  # weight=model.class_weights)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs channels
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    # 遍历每个预测输出
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            # 取出对应位置预测值
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression GIOU
            pxy = ps[:, :2].sigmoid() * 2. - 0.5                        # xy offset
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]          # 其没有采用exp操作，而是直接乘上anchors[i]
            pbox = torch.cat((pxy, pwh), 1)                             # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()                                  # iou loss

            # Objectness 有物体的conf分支权重
            # 类似fcos和yolov2，虽然我们引入了大量正样本anchor，但是不同anchor和gt bbox匹配度是不一样，预测框和gt bbox 的匹配度也不一样，如果权重设置一样肯定不是最优的，
            # 故作者将预测框和bbox的giou作为权重乘到conf分支，用于表征预测质量。
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)   # 建立预测targets的所有类别的大小
                t[range(n), tcls[i]] = cp                           # 对应预测目标位置置为1
                lcls += BCEcls(ps[:, 5:], t)                        # BCE 每个类单独计算loss

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]               # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

"""
核心操作
其和常规的yolov3 loss完全不同，其label没有跨层，对于任何一个bbox，三个输出层都有
对于任何一预测层，将每个bbox复制和anchor个数一样多的数目，然后当前bbox和当前层anchor一一对应计算匹配程度，算法不再是iou，而是shape比例
如果anchor和bbox的宽高比差距大于4，也就是认为不匹配，此时暂时把bbox删除，其实就相当于当做背景了
然后在对bbox计算落在那个网格，也就是说对于某个bbox落在的网格内部(注意此时落在网格也不再是一个，而是附近的多个，对原始中心点网格坐标扩展两个邻居像素，增加正样本数)，所有anchor都算Loss
前面shape过滤时候是不考虑bbox的xy坐标的，也就是说bbox的wh是和所有anchor匹配的，会导致找到的邻居也相当于进行了shape过滤规则
不存在取最大iou对应的anchor计算loss的设置，所以可能存在有些bbox在三个都预测的情况，也没有conf分支忽略阈值的操作
"""
def build_targets(p, targets, model):
    # 1.将targets 重复3遍(3=层anchor数目)，也就是将每个gt bbox复制变成独立的3份，方便和每个位置的3个anchor单独匹配
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, number of targets
    tcls, tbox, indices, anch, ttar = [], [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    # anchor=3个数,将target变成3*target格式，方便后面计算loss
    # anchor索引，后面有用，用于表示当前bbox和当前层的哪个anchor匹配
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    # 先repeat和当前层anchor个数一样，相当于每个bbox变成了3个，然后和3个anchor单独进行匹配
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias     # 网格中心偏移
    # 附近的4个网格
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):  # 3个输出分支
        anchors = det.anchors[i]  # 当前分支anchor
        # p是网络输出值，  1 1 特征图大小 特征图大小 特征图大小 特征图大小 1
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

        # 2.对每个输出层单独匹配。首先将targets变成anchor尺度，方便计算；然后将target wh shape和anchor的wh计算比例，如果其最大比例过大，则说明匹配度不高，将该bbox过滤，在当前层认为是bg
        # Match targets to anchors   targets的xywh本身是归一化尺度，故需要变成特征图尺度
        t = targets * gain
        if nt:
            # 计算当前target的wh和anchor的wh之间的比值
            # 如果w和h的最大比例大于预设阈值model.hyp['anchor_t']=4，则说明当前target和anchor匹配度不高，不应该强制回归，应该把target丢弃
            # 主要是把shape和anchor匹配度不高的label去掉，这其实也说明了该物体的大小比较极端，要么太大，要么太小，要么wh差距很大
            # 基于shape过滤后，就会出现某些bbox仅仅和当前层的几个anchor匹配，即可能出现某些bbox仅仅和其中某个匹配，而不是和当前位置的所有anchor匹配
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio		不考虑xy坐标
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter 注意过滤规则是没有考虑xy的，也就是当前bbox的wh是和所有anchor计算的

            # 3.计算最近的2个邻居网格
            # https://www.kaggle.com/c/global-wheat-detection/discussion/172436
            # 网格的3个附近点，不再是落在哪个网格就计算该网格的anchor，而是依靠中心点的情况
            # 选择最近的3个网格，作为落脚点，可以极大增加正样本数
            # 也就是对于保留的bbox，最少有3个anchor匹配，最多9个
            gxy = t[:, 2:4]  # grid xy  label的中心点坐标
            gxi = gain[[2, 3]] - gxy  # inverse
            # 这两个条件可以选择出最靠近的2个邻居，再加上自己，就是3个
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            # 5是因为预设的off是5个，现在选择出最近的3个（包括0, 0也就是自己）
            t = t.repeat((5, 1, 1))[j]  # (label个数x3,7) 附近的2个网格anchor,都算该bbox的anchor点
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 选择出最近的3个
        else:
            t = targets[0]
            offsets = 0

        # 4.对每个bbox找出对应的正样本anchor，其中包括b表示当前bbox属于batch内部的第几张图片，a表示当前bbox和当前层的第几个anchor匹配上，gi,gj是对应的负责预测该bbox的网格坐标，
        # gxy是不考虑offset或者说yolov3里面设定的该Bbox的负责预测网格，gwh是对应的归一化bbox wh，c是该Bbox类别
        # 按照yolov3,则直接（gxy - 0.5）.long()即可得到网格坐标
        # 但是这里考虑了附近网格，即采用了跨网格预测，估offsets不再是0.5而是2个邻居
        # 所以xy回归范围也变了，故xy预测输出不再是0-1，而是-1~1，加上offset偏移，则为-0.5-1.5
        # 由于shape过滤规则,宽高范围也也不再是任意范围，而是0-4，因为超过4倍比例是算不匹配的anchor,所以最大是4
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()  # 当前label落在哪个网格上
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices(j,i)
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        ttar.append(torch.cat((gxy, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch, ttar
