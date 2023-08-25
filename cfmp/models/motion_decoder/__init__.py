from mtr.models.motion_decoder.mtr_decoder import MTRDecoder
from cfm_decoder import MTRCFMDecoder

__all__ = {
    'MTRDecoder': MTRDecoder, 
    'CFMMTRDecoder': MTRCFMDecoder
}


def build_motion_decoder(in_channels, config):
    model = __all__[config.NAME](
        in_channels=in_channels,
        config=config
    )

    return model