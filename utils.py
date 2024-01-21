import cv2
import numpy as np

import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmdet.apis import set_random_seed
from mmdet.apis import init_detector
from algo2_result_to_aligned_result import algo2_result_to_aligned_result

from mmdet.models import build_detector

import subprocess
import random
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def print_LC(img_path, pred, out_path="../images/example_with_bounding_boxes.jpg"):
    img = cv2.imread(img_path)
    pred[np.isnan(pred)] = 0
    for box in pred:
        x0 = box[0]
        x1 = box[2]
        y0 = box[1]
        y1 = box[3]

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(img, start_point, end_point, color=(255, 50, 255), thickness=2)

        cv2.putText(
            img,
            "|".join(map(str, map(int, box[4:]))),
            (int(x0) + 15, int(y0) + 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(255, 50, 255),
            thickness=1,
        )

    cv2.imwrite(out_path, img)


def detect_quads(
    img_path,
    model=None,
    checkpoint_file="/media/quadro/NVME/Mehrab/exps/32_quad_long_retry/latest.pth",
    config_file="/media/quadro/NVME/Mehrab/Current_Experiment/config.py",
    numpy=True,
    integer=True,
):
    if model == None:
        cfg = Config.fromfile(config_file)

        set_random_seed(0, deterministic=False)
        model = build_detector(cfg.model, train_cfg=cfg.model.test_cfg)
        model.CLASSES = ("box",)

        model = init_detector(config_file, checkpoint_file,device='cuda:2')
    imgs = img_path
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
    data["img"] = [img.data[0] for img in data["img"]]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(m, RoIPool), "CPU inference with RoIPool is not supported currently."

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    # center_heatmap_preds, offset_preds, center2vertex_pred, vertex2center_pred = model.bbox_head(
    #     model.extract_feat(data["img"][0])
    # )
    # result_list = []
    # for img_id in range(len(data["img_metas"])):
    #     result_list.append(
    #         model.bbox_head._get_bboxes_single(
    #             center_heatmap_preds[0][img_id : img_id + 1, 0:1, ...],
    #             center2vertex_pred[0][img_id : img_id + 1, ...],
    #             offset_preds[0][img_id : img_id + 1, ...],
    #             data["img_metas"][img_id][0],
    #             rescale=False,
    #             with_nms=False,
    #         )
    #     )
    # quads = result_list[0][0]
    quads = results[0]
    # if numpy:
    #     quads = quads.cpu().detach().numpy()
    if integer:
        #quads = quads.astype(int)
        quads = [quad.astype(int) for quad in quads]
    return quads


def heads_output(
    img_path,
    model=None,
    checkpoint_file="/media/quadro/NVME/Mehrab/exps/32_quad_long_retry/latest.pth",
    config_file="/media/quadro/NVME/Mehrab/Current_Experiment/config.py",
):
    """
    center_heatmap_preds, offset_preds, c2v_pred, v2c_pred
    """
    if model == None:
        cfg = Config.fromfile(config_file)

        set_random_seed(0, deterministic=False)
        model = build_detector(cfg.model, train_cfg=cfg.model.test_cfg)
        model.CLASSES = ("box",)

        model = init_detector(config_file, checkpoint_file, device="cuda:2")
    imgs = img_path
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
    data["img"] = [img.data[0] for img in data["img"]]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(m, RoIPool), "CPU inference with RoIPool is not supported currently."

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    # model.forward(data['img'], data['img_metas'], return_loss=False)
    return model.bbox_head(model.extract_feat(data["img"][0]))
