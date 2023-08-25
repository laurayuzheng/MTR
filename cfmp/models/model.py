from mtr.models.model import MotionTransformer
from mtr.models.context_encoder import build_context_encoder
from .motion_decoder import build_motion_decoder

class CFMMotionTransformer(MotionTransformer):
    
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER)
        self.motion_decoder = build_motion_decoder(
            in_channels=self.context_encoder.num_out_channels,
            config=self.model_cfg.MOTION_DECODER
        )
    
    