from omegaconf import DictConfig, OmegaConf
import sys
from torch_points3d.applications.modelfactory import ModelFactory
print(OmegaConf.load("/d/gastruc/Desktop/torch-points3d/torch_points3d/applications/conf/pointnet2/encoder_3_ms.yaml"))
model_config=OmegaConf.load("/d/gastruc/Desktop/torch-points3d/torch_points3d/applications/conf/pointnet2/encoder_3_ms.yaml")
ModelFactory.resolve_model(model_config, 10)
print( sys.modules[__name__])