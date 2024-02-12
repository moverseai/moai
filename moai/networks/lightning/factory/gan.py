from moai.networks.lightning.feedforward import _create_scheduling_block, _create_supervision_block, _create_processing_block, _create_validation_block, _create_optimization_block

import moai.utils.parsing.rtp as mirtp
import moai.networks.lightning as minet
import moai.nn.linear as mil

import torch
import inspect
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging
import toolz
import numpy as np

from collections import defaultdict

log = logging.getLogger(__name__)

__all__ = ["GenerativeAdversarialNetwork"]

class GenerativeAdversarialNetwork(minet.FeedForward):
    def __init__(self, 
        # configuration:  omegaconf.DictConfig,
        io:             omegaconf.DictConfig,
        modules:        omegaconf.DictConfig,
        data:           omegaconf.DictConfig=None,
        parameters:     omegaconf.DictConfig=None,
        feedforward:    omegaconf.DictConfig=None,        
        monads:         omegaconf.DictConfig=None,        
        supervision:    omegaconf.DictConfig=None,
        validation:     omegaconf.DictConfig=None,
        generation:     omegaconf.DictConfig=None,
        visualization:  omegaconf.DictConfig=None,
        export:         omegaconf.DictConfig=None,
    ):
        super(GenerativeAdversarialNetwork, self).__init__(
            feedforward=feedforward, monads=monads, 
            visualization=visualization, data=data,
            supervision=supervision, validation=validation,
            export=export, parameters=parameters,
        )
        self.generator = hyu.instantiate(modules['generator'])
        self.discriminator = hyu.instantiate(modules['discriminator'])
        self.parameter_selectors = parameters.selectors or []
        self.scheduler_configs = toolz.get_in(['optimization', 'schedulers'], parameters) or { }
        self.params_optimizers = []
        self.stages = []
        self.optimizer_configs = toolz.get_in(['optimization', 'optimizers'], parameters) or []
        scheme_elements = toolz.get_in(['optimization', 'scheme'], parameters) or { }
        optimization = { }
        for element in scheme_elements:
            new_dict = toolz.get_in([element], scheme_elements)
            optimization = toolz.dicttoolz.merge(optimization, new_dict)
        log.info(f"GAN is optimized in {len(optimization)} stages.")
        for stage, cfg in optimization.items():
            self.stages.append(stage)
            log.info(
                f"Setting up the '{stage}' stage using the " + str(cfg.optimizer or "same") + " optimizer, "
                "optimizing the " + str(cfg.selectors or "same") + " parameters"
            )
            optimizer = cfg.optimizer
            frequency = cfg.frequency
            scheduler = cfg.scheduler
            selector = cfg.selectors
            self.params_optimizers.append((optimizer, stage, frequency, selector, scheduler))
            objective = cfg.objective
            self.supervision[stage] = _create_supervision_block(
                omegaconf.OmegaConf.merge(supervision, objective)
            )       
        self.sample = _create_processing_block(generation, "sample", monads=monads)
        self.walk = _create_processing_block(generation, "walk", monads=monads)
        self.generate = _create_processing_block(generation, "generate", monads=monads)
        self.gen_validation = _create_validation_block(toolz.get('generation',validation, None))
        self.gen_fwds, self.disc_fwds = [], []

        params = inspect.signature(self.generator.forward).parameters
        gen_in = list(zip(*[mirtp.force_list(io.generator[prop]) for prop in params]))
        gen_out = mirtp.split_as(mirtp.resolve_io_config(io.generator['out']), gen_in)

        self.gen_res_fill = [mirtp.get_result_fillers(self.generator, out) for out in gen_out]
        get_gen_filler = iter(self.gen_res_fill)
        
        for keys in gen_in:
            self.gen_fwds.append(lambda td,
                tk=keys,
                args=params.keys(), 
                gen=self.generator,
                filler=next(get_gen_filler):
                    filler(td, gen(**dict(zip(args,
                        list(
                                td[k] if type(k) is str
                                else list(td[j] for j in k)
                            for k in tk
                        )
                    ))))
            )

        params = inspect.signature(self.discriminator.forward).parameters
        disc_in = list(zip(*[mirtp.force_list(io.discriminator[prop]) for prop in params]))
        disc_out = mirtp.split_as(mirtp.resolve_io_config(io.discriminator['out']), disc_in)

        self.disc_res_fill = [mirtp.get_result_fillers(self.discriminator, out) for out in disc_out]
        get_disc_filler = iter(self.disc_res_fill)
        
        for keys in disc_in:
            self.disc_fwds.append(lambda td,
                tk=keys,
                args=params.keys(), 
                disc=self.discriminator,
                filler=next(get_disc_filler):
                    filler(td, disc(**dict(zip(args,
                        list(
                                td[k] if type(k) is str
                                else list(td[j] for j in k)
                            for k in tk
                        )
                    ))))
            )

    def forward(self, 
        td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for g in self.gen_fwds:
            g(td)
        return td

    def training_step(self, 
        batch:                  typing.Dict[str, torch.Tensor],
        batch_idx:              int,
        optimizer_idx:          int,
    ) -> typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]:
        preprocessed = self.preprocess(batch)
        prediction = self(preprocessed)
        for d in self.disc_fwds: #real
            d(prediction)
        postprocessed = self.postprocess(prediction)
        total_loss, losses = self.supervision[self.stages[optimizer_idx]](postprocessed)
        losses = toolz.keymap(lambda k: f"train_{k}", losses)
        losses.update({'total_loss': total_loss})        
        self.log_dict(losses, prog_bar=False, logger=True)        
        return { 'loss': total_loss, 'tensors': postprocessed }

    def configure_optimizers(self) -> typing.Tuple[typing.List[torch.optim.Optimizer], typing.List[torch.optim.lr_scheduler._LRScheduler]]:
        log.info("Configuring optimizer and scheduler")
        optimizers, schedulers, optim_list = [], [], []
        for optimizer, stage, frequency, params, schedule in self.params_optimizers:
            parameters = hyu.instantiate(self.parameter_selectors[params])(self) if isinstance(params, str) else\
                    list(hyu.instantiate(self.parameter_selectors[p])(self) for p in params)
                    # list(toolz.concat(hyu.instantiate(self.parameter_selectors[p])(self) for p in params))                    
            #TODO: parameter construction is very slow
            optimizers.append(_create_optimization_block(
                self.optimizer_configs[optimizer], parameters['params']
            ))#TODO: check if it works with a list of parameters
            setattr(optimizers[-1], 'frequency', [frequency])
            setattr(optimizers[-1], 'name', [params])
            setattr(optimizers[-1], 'stage', stage)
            schedulers.append(_create_scheduling_block(
                self.scheduler_configs.get(schedule, None), [optimizers[-1]]
            ))
            # if any(p in (self.assigned_params or []) for p in params):
            #     optimizers[-1].assign = self.assign
            #     self.prediction_stages.append(name)
        [optim_list.append({
                "optimizer": optimizers[idx],
                "scheduler": schedule[idx],
                "frequency": optimizers[idx].frequency[0]
            }
        ) for idx in range(len(optimizers))]
        return optim_list