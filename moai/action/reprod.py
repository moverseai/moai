from moai import __version__ as miV

import hydra
import sys
import os
import omegaconf.omegaconf
import logging
import typing
import ipaddress
from rich.console import Console

log = logging.getLogger(__name__)

def assign(cfg: omegaconf.DictConfig, attr: str) -> typing.Union[typing.Any,typing.Any]:
    return getattr(cfg, attr) if hasattr(cfg, attr) else None

def isIP(ip_address: str):
    ip_address = ip_address.replace('localhost', '127.0.0.1')
    try:
        ipaddress.ip_address(ip_address)
        return True
    except:
        pass

def search_obj(obj: omegaconf.DictConfig, console: Console, var_path=[]):
    for k, v in obj.items():
        var_path.append(k)
        if not isinstance(v, omegaconf.DictConfig):
            if isinstance(v, str):
                values = os.path.splitext(v)
                if k == 'gpus':
                    full_key = '.'.join(var_path)
                    value = ''
                    while not (
                        (len(value) == 1 and value.isdigit()) or \
                        (len(value) == 2 and value[1:].isdigit() and int(value) > -2 and int(value) < 10)) \
                        and not value == 's': # NOTE: to check the conditions for GPUs values
                        value = console.input(f"==> {full_key} ([italic cyan]{obj[k]}[/])? [[bold yellow]s (skip)[/]] ").lower()
                    if not value=='s':
                        obj[k] = value
                        console.print(f"[green]Param set[/] --> ([italic cyan]{obj[k]}[/])")
                    else:
                        console.print(f"[yellow]Param skipped[/] --> ([italic cyan]{obj[k]}[/])")
                elif len(os.path.dirname(values[0])):
                    full_key = '.'.join(var_path)
                    value = ''
                    while not len(os.path.dirname(value)) and not value == 's':
                        value = console.input(f"==> {full_key} ([italic cyan]{obj[k]}[/])? [[bold yellow]s (skip)[/]] ").lower()
                    if not value=='s':
                        obj[k] = value
                        console.print(f"[green]Param set[/] --> ([italic cyan]{obj[k]}[/])")
                    else:
                        console.print(f"[yellow]Param skipped[/] --> ([italic cyan]{obj[k]}[/])")
                else:
                    if isIP(v):
                        full_key = '.'.join(var_path)
                        value = ''
                        while not (isinstance(value, str) and isIP(value)) and not value == 's':
                            value = console.input(f"==> {full_key} ([italic cyan]{obj[k]}[/])? [[bold yellow]s (skip)[/]] ").lower()
                        if not value=='s':
                            obj[k] = value
                            console.print(f"[green]Param set[/] --> ([italic cyan]{obj[k]}[/])")
                        else:
                            console.print(f"[yellow]Param skipped[/] --> ([italic cyan]{obj[k]}[/])")
            var_path.pop(-1)
        else:
            search_obj(v, console, var_path)
    if len(var_path):
        var_path.pop(-1)

def env_query(cfg: omegaconf.DictConfig, console: Console):
    command = console.input("[bold yellow]Is the environment the one where the experiment has been executed? (y/n) [/]").lower()
    if command == "y":
        pass
    elif command == "n":
        search_obj(cfg, console)
    else:
        env_query(cfg, console)

def reprod(cfg):    
    console = Console()
    console.rule("Environment Settings", style="bold blue")
    with omegaconf.open_dict(cfg):
        del cfg["reprod"]
        env_query(cfg, console)
    console.rule("Reproduction Begin", style="bold blue")
    hydra.utils.log.debug(f"Configuration:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    with open("config_resolved.yaml", 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    with open("config.yaml", 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    engine = hydra.utils.instantiate(cfg.engine)
    model = hydra.utils.instantiate(cfg.model,
        data=cfg.data,
        visualization=assign(cfg, "visualization"),
        export=assign(cfg, "export"),    
    )
    model.hparams.update(omegaconf.OmegaConf.to_container(cfg, resolve=True))
    model.hparams['__moai__'] = { 'version': miV }
    for name, remodel in (assign(cfg, "remodel") or {}).items():
        hydra.utils.instantiate(remodel)(model)
    model.initialize_parameters()
    trainer = hydra.utils.instantiate(cfg.trainer, 
        logging=assign(cfg, "logging")
    )
    log.info("Training started.")     
    trainer.run(model)
    log.info("Training completed.")

if __name__ == "__main__":
    # os.environ['HYDRA_FULL_ERROR'] = '1'
    config_filename = get_conf_path(sys.argv.pop(1)) #TODO: argparser integration?
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${experiment.name}")    
    reprod = hydra.main(config_path="conf", config_name=config_filename)(reprod)
    reprod()