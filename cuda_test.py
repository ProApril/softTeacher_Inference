import mmcv
from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot
from ssod.apis.inference import init_detector, save_result
from ssod.utils import patch_config

#export OMP_NUM_THREADS=1

fconfig = "configs/soft_teacher/soft_teacher_faster_rcnn_r101_caffe_fpn_coco_full_1080k.py"
fcheckpoint = "work_dirs/iter_1080000.pth"
#OPT_DEVICE = opt_device
OPT_DEVICE = "cuda:0"
#OPT_DEVICE = "cpu"

# bbox score threshold
cfg = Config.fromfile(fconfig)

# Not affect anything, just avoid index error
cfg.work_dir = "./work_dirs"
cfg = patch_config(cfg)
# build the model from a config file and a checkpoint file
model = init_detector(cfg, fcheckpoint, device=OPT_DEVICE)