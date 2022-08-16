from cmath import e
import hydra
import sys
import omegaconf.omegaconf
import typing
import logging
import pytorch_lightning as pl
import blobconverter
from moai.utils.torch import get_submodule
import torch
import os

log = logging.getLogger(__name__)


class TraceWrapper(pl.LightningModule):
    def __init__(self,
        module_names: typing.List[str], #exported module names
        output_names: typing.List[str], #output names
        input_names: typing.List[str], #input names
        model: torch.nn.Module,
    ) -> None:
        
        super().__init__()

        
        self.model = model
        self.exported_module_names = module_names
        self.input_names = input_names
        self.output_names = output_names

    
    def forward(self,
        x: typing.Dict[str, torch.Tensor],
    ) -> typing.Dict[str, torch.Tensor]:
        #do something
        for module in self.exported_module_names:
            try:
                m = get_submodule(self, module) if module != "model" \
                    else self.model
            except:
                log.error(f'The {m} module does not exist in your model!')
                log.info(f'Please select on of the following {*self.model._modules.keys(),}')
            x = m(x)
        return {out_key: x[out_key] for out_key in self.output_names}




def export(cfg):
    hydra.utils.log.debug(f"Configuration:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    with open("config_resolved.yaml", 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    with open("config.yaml", 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    engine = hydra.utils.instantiate(cfg.engine)
    model = hydra.utils.instantiate(cfg.model)
    #TODO: add cpu device support
    device = cfg.tester.gpus[0] if 'tester' in cfg.keys() \
        else cfg.trainer.gpus[0]

    model.eval()
    model.initialize_parameters()
    model.to(device)


    trwrapper = TraceWrapper(
        cfg.export.module_names,
        cfg.export.output_names,
        cfg.export.input_names,
        model
    )

    trwrapper.eval()
    trwrapper.to(device)
    
    input_dict = {}

    for in_name in cfg.export.input_names:
        input_dict[in_name] = torch.randn(tuple(cfg.export.input_tensor[in_name])).to(device).float()

    if cfg.export.mode == 'onnx':
        log.info("exporting model to onnx!")
        trwrapper.to_onnx(
            os.path.join(cfg.export.output_path, f'{cfg.export.name}.onnx'),
            # {trwrapper.input_names[0]: input_sample},
            input_dict, 
            export_params=cfg.export.export_params, #bool 
            opset_version=cfg.export.opset_version, #int; default is 12
            input_names=trwrapper.input_names,
            output_names=trwrapper.output_names,
        )
    
    elif cfg.export.mode == 'jit':
        #TODO: Add trace support
        #trace
        log.info("converting model to be jit compatible!")
        traced_model = torch.jit.trace(
            trwrapper, #model 
            {trwrapper.input_names[0]: input_sample}, #example inputs
            strict=False, # to allow mutable (e.g. dicts) output
            )
        # traced_model = torch.jit.trace(
        #     t, #model 
        #     {t.model_in[0][0]: input_sample}, #example inputs
        #     strict=False, # to allow mutable (e.g. dicts) output
        #     )
        traced_model.save(
            os.path.join(cfg.export.output_path, f'{cfg.export.name}.pt')
        ) # Save
    
    elif cfg.export.mode == 'blob':
        #first convert to onnx
        log.info("exporting model to onnx!")
        trwrapper.to_onnx(
            os.path.join(cfg.export.output_path, f'{cfg.export.name}.onnx'),
            # {trwrapper.input_names[0]: input_sample},
            input_dict, 
            export_params=cfg.export.export_params, #bool 
            opset_version=cfg.export.opset_version, #int; default is 12
            input_names=trwrapper.input_names,
            output_names=trwrapper.output_names,
            verbose=True,
        )
        blob_path=blobconverter.from_onnx(
            model=os.path.join(cfg.export.output_path, f'{cfg.export.name}.onnx'),
            data_type=cfg.export.data_type,
            shaves=cfg.export.shaves,
            output_dir=cfg.export.output_path
        )
        #delete onnx file?


    log.info(f"Model has been saved in {cfg.export.output_path}!")


if __name__ == "__main__":  
    config_filename = sys.argv.pop(1) #TODO: argparser integration?
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${export.name}")
    export = hydra.main(config_path="conf", config_name=config_filename)(export)
    export()