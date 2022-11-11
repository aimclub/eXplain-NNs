import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
from fasterRCNN import decompose_module, prune_model
import copy


def load_baseline(model_fldr:Path=None):
    device = torch.device('cpu')
    if path is None: path = Path('.')
    model_name = "ResNet50.sd.pt"
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 50)
    model.load_state_dict(torch.load(model_fldr/model_name,  map_location=device))
    return model

def load_improved(model_fldr:Path=None):
    svd_model_name = "ResNet50_SVD_channel_O-10.0_H-0.001000.sd.pt"
    svd_model = fasterrcnn_resnet50_fpn()
    in_features = svd_model.roi_heads.box_predictor.cls_score.in_features
    svd_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 50)
    decompose_module(svd_model.backbone, "channel")
    svd_model.load_state_dict(torch.load(model_fldr/svd_model_name,  map_location=device))
    return svd_model

def get_pruned_model(model_fldr:Path=None):
    svd_model = load_improved(model_fldr)
    pruned_model = copy.deepcopy(svd_model)
    prune_model(model=pruned_model.backbone, energy_threshold=0.9)
    return pruned_model