import torch
import quix
import torch.nn as nn
from model import DifferentiableSuperpixelTokenizerViT
from numbers import Number
from typing import Optional
from quix import AbstractLogger

'''Run Example Usage:

samsida: torchrun --standalone --nnodes 1 --nproc_per_node 4 train.py --cfgfile ./cfg/B16_samsida.toml
'''

_architecture_cfg = {
    'T': {'depth':12, 'embed_dim': 192, 'heads': 3, 'dop_path':0.0},
    'S': {'depth':12, 'embed_dim': 384, 'heads': 6, 'dop_path':0.1},
    'M': {'depth':12, 'embed_dim': 512, 'heads': 8, 'dop_path':0.1},
    'B': {'depth':12, 'embed_dim': 768, 'heads':12, 'dop_path':0.2},
    'L': {'depth':24, 'embed_dim':1024, 'heads':16, 'dop_path':0.2},
    'H': {'depth':32, 'embed_dim':1280, 'heads':16, 'dop_path':0.2},
}


# (self, model_name, max_segments, num_classes, num_channels, superpixel_algorithm='voronoi_propagation', pretrained=False)

class DifferentiableSuperpixelTokenizerViTModelConfig(quix.ModelConfig):
    '''GaMBiT ModelConfig

    Attributes
    ----------
    model_name : str
        Name for model configuration in timm.
    n_segments : int
        The number of segments for the tokenizer.
    n_channels : int
        Number of channels for input image.
    suerpixel_algorithm : float
        Str for which superpixel algorithm to use for mask generation. Options: ['vornoi_propagation', 'SLIC_segmentation', 'Boundary_Pathfinder'].
    pretrained : bool
        Flag for using pretrained timm weights for the ViT.
    '''
    n_segments:int = 196
    n_channels:int = 3
    suerpixel_algorithm:str ='voronoi_propagation'
    pretrained:bool = False


class DifferentiableSuperpixelTokenizerViTOptimizerConfig(quix.OptimizerConfig):
    '''GaMBiT OptimizerConfig

    Attributes
    ----------
    loss_fn : str
        Loss function to use for training.
    '''
    loss_fn:str = quix.cfg.add_argument(
        default='mse', choices=['mse', 'logcosh']
    )


class nSigmaLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['n_sigma'])


class LogpriorLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['logprior'])


class DifferentiableSuperpixelTokenizerViTRunner(quix.Runner):

    @property
    def mod(self) -> DifferentiableSuperpixelTokenizerViTModelConfig:
        return self.cfg.mod # type: ignore
    
    @property
    def opt(self) -> DifferentiableSuperpixelTokenizerViTModelConfig:
        return self.cfg.opt # type: ignore
    
    def parse_model(self):
        capacity = self.mod.model[0]
        ksize = int(self.mod.model[1:])
        if capacity in _architecture_cfg:
            modeldict = _architecture_cfg[capacity]
        else:
            raise ValueError(f'No valid architecture found for {self.mod.model}')
        modeldict['model_name'] = ksize
        modeldict['n_segments'] = self.mod.n_segments 
        modeldict['n_channels'] = self.mod.n_channels
        modeldict['superpixel_algorithm'] = self.mod.superpixel_algorithm
        modeldict['pretrained'] = self.mod.pretrained
        model = DifferentiableSuperpixelTokenizerViT(**modeldict)
        return model

    def parse_logger(self):
        rank = local_rank = 0
        if self.distributed:
            if self.rank is not None:
                rank = self.rank
            if self.local_rank is not None:
                local_rank = self.local_rank
        loggers = [
            quix.ProgressLogger(), quix.DeltaTimeLogger(), quix.LossLogger(),
            nSigmaLogger(), LogpriorLogger(),
            quix.LRLogger(), quix.GPULogger()
        ]
        custom_runid = (
            self.__class__.__name__ + '_' + self.mod.model
            if self.log.custom_runid is None else self.log.custom_runid
        )        
        return quix.LogCollator(
            custom_runid,
            self.savedir,
            rank,
            local_rank,
            loggers,
            stdout=self.log.stdout
        )
    
    def unpack_data(self, data):
        return (data,), None
    
    def parse_loss(self):
        match self.opt.loss_fn:
            case 'mse':
                return nn.MSELoss()
            case 'logcosh':
                return gambit.LogCosh()
            case _:
                raise ValueError(f'Unknown argument: {self.opt.loss_fn}')
    
    @staticmethod
    def forward_fn(inputs, _, model, loss_fn):
        xh, x = model(*inputs)
        loss = loss_fn(xh, x)
        n_sigma = model.module.encoder.tokenizer.n_sigma.item()
        logprior = model.module.encoder.tokenizer.logprior.item()
        return {'outputs':None, 'loss':loss, 'n_sigma':n_sigma, 'logprior':logprior}

    
if __name__ == '__main__':
    runcfg = quix.RunConfig.argparse(modcfg=DifferentiableSuperpixelTokenizerViTModelConfig, optcfg=DifferentiableSuperpixelTokenizerViTOptimizerConfig)
    DifferentiableSuperpixelTokenizerViTRunner(runcfg).run()
