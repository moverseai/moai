from moai.utils.arguments import ensure_string_list

import moai.networks.lightning as minet
import moai.nn.convolution as mic
import moai.nn.residual as mires
import moai.nn.sampling.spatial.downsample as mids
import moai.modules.lightning as mimod
import moai.nn.utils as miu

import torch

import hydra.utils as hyu
import omegaconf.omegaconf as omegaconf
import typing
import logging

log = logging.getLogger(__name__)

#NOTE: from https://github.com/HRNet/HRNet-Bottom-Up-Pose-Estimation/blob/master/lib/models/pose_hrnet.py
#NOTE: from https://arxiv.org/pdf/1908.07919.pdf

__all__ = ["HRNet"]

class HRNet(minet.FeedForward):
    def __init__(self,
        configuration:  omegaconf.DictConfig,
        modules:        omegaconf.DictConfig,
        data:           omegaconf.DictConfig=None,
        parameters:     omegaconf.DictConfig=None,
        feedforward:    omegaconf.DictConfig=None,
        monads:         omegaconf.DictConfig=None,
        supervision:    omegaconf.DictConfig=None,
        validation:     omegaconf.DictConfig=None,
        visualization:  omegaconf.DictConfig=None,
        export:         omegaconf.DictConfig=None,
    ):
        super(HRNet, self).__init__(
            data=data, parameters=parameters,
            feedforward=feedforward, monads=monads,
            supervision=supervision, validation=validation,
            export=export, visualization=visualization,            
        )
        preproc = configuration.preproc
        #NOTE: preproc = stem + layer1
        preproc_convs = []
        prev_features = configuration.in_features
        stem = preproc.stem
        for b, c, a, f, k, s, p in zip(
            stem.blocks, stem.convolutions,
            stem.activations, stem.features,
            stem.kernel_sizes, stem.strides, stem.paddings):
            preproc_convs.append(mic.make_conv_block(
                block_type=b,
                convolution_type=c,
                in_features=prev_features, 
                out_features=f,
                activation_type=a,
                convolution_params={
                    "kernel_size": k,
                    "stride": s,
                    "padding": p,
                },
            ))
            prev_features = f
        residual = preproc.residual
        residual_blocks = []
        for i, o, b in zip(
            residual.features.in_features, residual.features.out_features,
            residual.features.bottleneck_features, 
        ):
            residual_blocks.append(mires.make_residual_block(
                block_type=residual.block,
                convolution_type=residual.convolution,
                out_features=o,
                in_features=i,
                bottleneck_features=b,
                activation_type=residual.activation,
                strided=False,
            ))
        self.pre = torch.nn.Sequential(
            *preproc_convs, *residual_blocks,
        )
        branches_config = configuration.branches
        start_trans_config = modules['start_transition']
        self.start_trans = hyu.instantiate(start_trans_config, 
            in_features=residual.features.out_features[-1],
            start_features=branches_config.start_features
        )
        #NOTE: stages
        highres_module = modules['highres'] # NOTE: outputs list of # branches outputs
        self.stages = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                hyu.instantiate(highres_module, 
                    branches=i, depth=d, start_features=branches_config.start_features
                ) for _, d in zip(range(modules), depths)
            ]) for i, modules, depths in zip(
                range(2, configuration.stages + 1),
                branches_config.modules,
                branches_config.depths,
            )
        ])
        stage_trans_config = modules['stage_transition']
        self.stage_transitions = torch.nn.ModuleList([
            hyu.instantiate(stage_trans_config, branches=i + 1,
                prev_branch_features=branches_config.start_features * (2 ** i),
            ) for i in range(1, configuration.stages - 1)
        ])
        head_module = modules['head']
        self.head = hyu.instantiate(head_module,
            stages=configuration.stages,
            start_features=branches_config.start_features,
            out_features=configuration.out_features,
        )
        self.input = ensure_string_list(configuration.input)
        self.output = ensure_string_list(configuration.output)
        self.output_prefix = configuration.output

    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:        
        for i, o in zip(self.input, self.output):
            in_tensor = td[i]
            preproc = self.pre(in_tensor)
            hr_inputs = self.start_trans(preproc)
            for stage, trans in zip(self.stages, self.stage_transitions):
                hr_inputs = trans(stage(hr_inputs))
            prediction = self.head(self.stages[-1](hr_inputs))
            if type(prediction) == list: #NOTE: to support higherhnet                
                for i , heatmap in enumerate(prediction):
                    td[f"{self.output_prefix[:-2]}_{i+1}"] = heatmap
            else:
                td[o] = prediction
        return td