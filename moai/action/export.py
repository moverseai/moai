from moai.utils.torch import get_submodule

import hydra
import sys
import omegaconf.omegaconf
import typing
import logging
import pytorch_lightning as pl
import torch
import os
import onnx
import toolz
from subprocess import check_output
from pathlib import Path
from subprocess import CalledProcessError


log = logging.getLogger(__name__)

try:
    import blobconverter
except:
    log.warning("The `blobconverter` package was not found, exporting to OpenVivo is disabled.")

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
        # get moai git info
        try:
            moai_url = get_command_output(["git", "ls-remote", '--get-url', 'origin'],  Path(sys.argv[0]).parent, strip=True)
            moai_commit = get_command_output(['git', 'rev-parse', 'HEAD'],  Path(sys.argv[0]).parent, strip=True)
        except (CalledProcessError, UnicodeDecodeError) as ex:
            log.warning(
                "Can't get information for repo in {}: {}".format(path, str(ex))
            )     
        project_path_urls = []
        project_path_commits = []
        if cfg.project_path:
            for path in cfg.project_path:
                try:
                    project_path_urls.append(
                        get_command_output(["git", "ls-remote", '--get-url', 'origin'], path, strip=True)
                    )
                    project_path_commits.append(
                        get_command_output(['git', 'rev-parse', 'HEAD'], path, strip=True)
                    )
                except:
                    log.warning(
                        "Can't get information for repo in {}: {}".format(path, str(ex))
                    )
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
        # Adding metadata
        model = onnx.load(os.path.join(cfg.export.output_path, f'{cfg.export.name}.onnx'))
        if moai_url: # add moai repo details
            model.metadata_props.append(onnx.StringStringEntryProto(key='moai', value=str(moai_url)))
            model.metadata_props.append(onnx.StringStringEntryProto(key='moai-commit', value=str(moai_commit)))
        if project_path_urls: # add repo details
            for url,commit in zip(project_path_urls,project_path_commits):
                model.metadata_props.append(onnx.StringStringEntryProto(key='repo', value=str(url)))
                model.metadata_props.append(onnx.StringStringEntryProto(key='repo-commit', value=str(commit)))
        try:
            for key in cfg.onnx.metadata:
                v = toolz.get_in(key.split("."),cfg)
                model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=str(v)))
        except:
            log.warning(f" Metadata has not been included in onnx eport.")
        try:
            for key in cfg.onnx.attributes:
                try:
                    model.__setattr__(key,cfg.onnx.attributes[key])
                except:
                    log.warning(f" Attribute {key} does not supported in existing onnx version.")
        except:
            log.warning(f" Attributes have not been included in onnx eport.")
        # Save the ONNX model
        onnx.save(model, os.path.join(cfg.export.output_path, f'{cfg.export.name}.onnx'))

    
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


def get_command_output(command, path=None, strip=True):
    """
    Run a command and return its output
    :raises CalledProcessError: when command execution fails
    :raises UnicodeDecodeError: when output decoding fails
    """
    with open(os.devnull, "wb") as devnull:
        result = check_output(command, cwd=path, stderr=devnull).decode()
        return result.strip() if strip else result

if __name__ == "__main__":  
    config_filename = sys.argv.pop(1) #TODO: argparser integration?
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${export.name}")
    export = hydra.main(config_path="conf", config_name=config_filename)(export)
    export()