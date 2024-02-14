from moai.parameters.initialization import Default as NoInit
from moai.networks.lightning.feedforward import _create_scheduling_block,\
                    _create_supervision_block,\
                    _create_processing_block,\
                    _create_validation_block,\
                    _create_optimization_block,\
                    _create_interval_block,\
                    _assign_data

from moai.data.iterator import Indexed
from collections import defaultdict

import moai.utils.parsing.rtp as mirtp
import pytorch_lightning

import torch
import inspect
import hydra.utils as hyu
import omegaconf.omegaconf
import typing
import logging
import toolz
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["GenerativeAdversarialNetwork"]

class GenerativeAdversarialNetwork(pytorch_lightning.LightningModule):
    def __init__(self, 
        # configuration:  omegaconf.DictConfig,
        io:             omegaconf.DictConfig,
        modules:        omegaconf.DictConfig,
        data:           omegaconf.DictConfig=None,
        parameters:     omegaconf.DictConfig=None,
        gan:            omegaconf.DictConfig=None,        
        monads:         omegaconf.DictConfig=None,        
        supervision:    omegaconf.DictConfig=None,
        validation:     omegaconf.DictConfig=None,
        visualization:  omegaconf.DictConfig=None,
        export:         omegaconf.DictConfig=None,
    ):
        super(GenerativeAdversarialNetwork, self).__init__()
        self.data = _assign_data(data)
        self.initializer = parameters.initialization if parameters is not None else None
        self.generator = hyu.instantiate(modules['generator'])
        self.discriminator = hyu.instantiate(modules['discriminator'])
        self.supervision = _create_supervision_block(supervision)
        self.validation = _create_validation_block(validation)
        self.visualization = _create_interval_block(visualization)
        self.exporter = _create_interval_block(export)
        self.parameter_selectors = parameters.selectors or []
        self.scheduler_configs = toolz.get_in(['optimization', 'schedulers'], parameters) or { }
        self.schedule_monitor = parameters.schedule_monitor if parameters is not None and parameters.schedule_monitor is not None else None
        self.params_optimizers = []
        self.stages = []
        self.optimizer_configs = toolz.get_in(['optimization', 'optimizers'], parameters) or []
        optimization = toolz.get_in(['optimization', 'process'], parameters) or { }
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
            opt_type = cfg.type
            self.params_optimizers.append((optimizer, opt_type, stage, frequency, selector, scheduler))
            objective = cfg.objective
            self.supervision[stage] = _create_supervision_block(
                omegaconf.OmegaConf.merge(supervision, objective)
            )
        self.preprocess = _create_processing_block(gan, "preprocess", monads=monads) 
        self.discriminate = _create_processing_block(gan, "discriminate", monads=monads)
        self.reconstruct = _create_processing_block(gan, "reconstruct", monads=monads)
        self.generate = _create_processing_block(gan, "generate", monads=monads)
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
                                td[k] if isinstance(k,str)
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
                                td[k] if isinstance(k,str)
                                else list(td[j] for j in k)
                            for k in tk
                        )
                    ))))
            )
    
    def initialize_parameters(self) -> None:
        init = hyu.instantiate(self.initializer) if self.initializer else NoInit()
        init(self)

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
        if 'reconstruct' in self.params_optimizers[optimizer_idx]:
            postprocessed = self.reconstruct(prediction)
            total_loss, losses = self.supervision[self.stages[optimizer_idx]](postprocessed)
        elif 'discriminate' in self.params_optimizers[optimizer_idx]:
            for d in self.disc_fwds:
                d(prediction)
            postprocessed = self.discriminate(prediction)
            total_loss, losses = self.supervision[self.stages[optimizer_idx]](postprocessed)
        else:
            for d in self.disc_fwds:
                d(prediction)
            postprocessed = self.generate(prediction)
            total_loss, losses = self.supervision[self.stages[optimizer_idx]](postprocessed)
        losses = toolz.keymap(lambda k: f"train_{k}", losses)
        losses.update({'total_loss': total_loss})        
        self.log_dict(losses, prog_bar=False, logger=True)        
        return { 'loss': total_loss, 'tensors': postprocessed }

    def training_step_end(self, 
        train_outputs: typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]]
    ) -> None:
        if self.global_step and (self.global_step % self.visualization.interval == 0):
            self.visualization(train_outputs['tensors'], self.global_step)
        if self.global_step and (self.global_step % self.exporter.interval == 0):
            self.exporter(train_outputs['tensors'], self.global_step)
        return train_outputs['loss']

    def validation_step(self,
        batch:              typing.Dict[str, torch.Tensor],
        batch_nb:           int,
        dataloader_index:   int=0,
    ) -> dict:        
        preprocessed = self.preprocess(batch)
        prediction = self(preprocessed)
        outputs = self.generate(prediction)
        #TODO: consider adding loss maps in the tensor dict
        metrics = self.validation(outputs)
        return metrics
    
    def validation_epoch_end(self,
        outputs: typing.Union[typing.List[typing.List[dict]], typing.List[dict]]
    ) -> None:
        list_of_outputs = [outputs] if isinstance(toolz.get([0, 0], outputs)[0], dict) else outputs
        all_metrics = defaultdict(list)
        for i, o in enumerate(list_of_outputs):
            keys = next(iter(o), { }).keys()        
            metrics = { }
            for key in keys:
                metrics[key] = np.mean(np.array(
                    [d[key].item() for d in o if key in d]
                ))
                all_metrics[key].append(metrics[key])            
            log_metrics = toolz.keymap(lambda k: f"val_{k}/{list(self.data.val.iterator.datasets.keys())[i]}", metrics)
            self.log_dict(log_metrics, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)
        all_metrics = toolz.valmap(lambda v: sum(v) / len(v), all_metrics)
        self.log_dict(all_metrics, prog_bar=True, logger=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> typing.Tuple[typing.List[torch.optim.Optimizer], typing.List[torch.optim.lr_scheduler._LRScheduler]]:
        log.info("Configuring optimizer and scheduler")
        optimizers, schedulers, optim_list = [], [], []
        for optimizer, opt_type, stage, frequency, params, schedule in self.params_optimizers:
            parameters = hyu.instantiate(self.parameter_selectors[params])(self) if isinstance(params, str) else\
                    list(hyu.instantiate(self.parameter_selectors[p])(self) for p in params)
            #TODO: parameter construction is very slow
            optimizers.append(_create_optimization_block(
                self.optimizer_configs[optimizer], parameters['params']
            ))#TODO: check if it works with a list of parameters
            setattr(optimizers[-1], 'frequency', [frequency])
            setattr(optimizers[-1], 'name', [params])
            setattr(optimizers[-1], 'stage', stage)
            setattr(optimizers[-1], 'type', opt_type)
            schedulers.append(_create_scheduling_block(
                self.scheduler_configs.get(schedule, None), [optimizers[-1]]
            ))
        [optim_list.append({
                "optimizer": optimizers[idx],
                "scheduler": schedule[idx],
                "frequency": optimizers[idx].frequency[0],
            }
        ) for idx in range(len(optimizers))]
        return optim_list
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.train.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.train.iterator._target_.split('.')[-1]}) train set data iterator")
            train_iterator = hyu.instantiate(self.data.train.iterator)
        else:
            train_iterator = Indexed(
                self.data.train.iterator.datasets,
                self.data.train.iterator.augmentation if hasattr(self.data.train.iterator, 'augmentation') else None,
            )
        if not hasattr(self.data.train, 'loader'):
            log.error("Train data loader missing. Please add a data loader (i.e. \'- data/train/loader: torch\') entry in the configuration.")
        else:
            train_loader = hyu.instantiate(self.data.train.loader, train_iterator)
        return train_loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.val.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.val.iterator._target_.split('.')[-1]}) validation set data iterator")
            val_iterators = [hyu.instantiate(self.data.val.iterator)]
        else:
            val_iterators = [Indexed(
                {k: v }, # self.data.val.iterator.datasets,
                self.data.val.iterator.augmentation if hasattr(self.data.val.iterator, 'augmentation') else None,
            ) for k, v in self.data.val.iterator.datasets.items()]
        if not hasattr(self.data.val, 'loader'):
            log.error("Validation data loader missing. Please add a data loader (i.e. \'- data/val/loader: torch\') entry in the configuration.")
        else:
            validation_loaders = [
                hyu.instantiate(self.data.val.loader, val_iterator)
                for val_iterator in val_iterators
            ]
        # return validation_loaders[0] if len(validation_loaders) == 1 else validation_loaders
        return validation_loaders

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if hasattr(self.data.test.iterator, '_target_'):
            log.info(f"Instantiating ({self.data.test.iterator._target_.split('.')[-1]}) test set data iterator")
            test_iterators = [hyu.instantiate(self.data.test.iterator)]
            #test_iterator = hyu.instantiate(self.data.test.iterator)
        else:
            test_iterators = [Indexed(
                {k: v }, # self.data.val.iterator.datasets,
                self.data.test.iterator.augmentation if hasattr(self.data.test.iterator, 'augmentation') else None,
            ) for k, v in self.data.test.iterator.datasets.items()]
            # test_iterator = Indexed(
            #     self.data.test.iterator.datasets,
            #     self.data.test.iterator.augmentation if hasattr(self.data.test.iterator, 'augmentation') else None,
            # )
        if not hasattr(self.data.test, 'loader'):
            log.error("Test data loader missing. Please add a data loader (i.e. \'- data/test/loader: torch\') entry in the configuration.")
        else:
            test_loaders = [
                hyu.instantiate(self.data.test.loader, test_iterator)
                for test_iterator in test_iterators
            ]
            #test_loader = hyu.instantiate(self.data.test.loader, test_iterator)
        return test_loaders