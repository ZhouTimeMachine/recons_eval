import torch
import torch.nn as nn

from typing import Optional, Dict

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class Pi3(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.pi3.models.pi3 import Pi3 as Pi3Model
            self.model = Pi3Model.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, images: torch.Tensor):
        return self.model(images)


class VGGT(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.vggt.models.vggt import VGGT as VGGTModel
            self.model = VGGTModel.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        return self.model(images, query_points)
    

class MoGe(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            ori_model: bool = True,
        ):
        super().__init__()

        if ori_model and pretrained_model_name_or_path is not None:
            from models.moge.model.v1 import MoGeModel
            self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        return self.model(image, num_tokens)


# class AetherV1(nn.Module):
#     def __init__(
#             self,
#             pretrained_model_name_or_path: Optional[str] = None,
#             ori_model: bool = True,
#         ):
#         super().__init__()

#         if ori_model and pretrained_model_name_or_path is not None:
#             from models.aether.v1 import AetherV1Model
#             self.model = AetherV1Model.from_pretrained(pretrained_model_name_or_path)
#         else:
#             raise NotImplementedError
#     def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
#         return self.model(images, query_points)

