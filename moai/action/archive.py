import moai.serve.model as model_server
import moai.serve.optimizer as optimizer_server

import yaml
import toolz
import sys
import hydra
import omegaconf.omegaconf
import logging
import subprocess
import typing
import glob
import os
import zipfile
import toolz

log = logging.getLogger(__name__)

def get_files(
    root:       str,
    folder:     typing.Union[str, typing.Sequence[str]],
    pattern:    str,
) -> typing.Sequence[str]:
    folders = [folder] if isinstance(folder, str) else folder
    folders = [os.path.join(root, f) for f in folders]
    files = []
    for f in folders:
        path = f if os.path.isabs(f) else os.path.join(hydra.utils.get_original_cwd(), f)        
        files += glob.glob(os.path.join(path, pattern))
        files += glob.glob(os.path.join(path, "**", pattern), recursive=True)
    return list(toolz.unique(files))

def dump_zipfile(
    filename:   str,
    root:       str,
    files:      typing.Sequence[str],
):
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as z:
        for file in files:                        
            arcname = os.path.join('.', file.replace(root, ''))
            z.write(file, arcname)

def dump_handlers(
    handlers:               omegaconf.omegaconf.DictConfig,
    handler_config:         omegaconf.omegaconf.DictConfig,
):
    empty = omegaconf.DictConfig({})
    with open('pre.yaml', 'w') as f:
        data = omegaconf.OmegaConf.to_container(handlers.get('pre') or empty)
        data = list(map(lambda e: {e.split(':')[0].strip(): e.split(':')[1].strip()}, data))
        yaml.dump({ 'defaults': data }, f, default_style=None, default_flow_style=False)
    with open('post.yaml', 'w') as f:
        data = omegaconf.OmegaConf.to_container(handlers.get('post') or empty)
        data = list(map(lambda e: {e.split(':')[0].strip(): e.split(':')[1].strip()}, data))
        yaml.dump({ 'defaults': data }, f, default_style=None, default_flow_style=False)
    if handler_config is not None:
        with open('handler_overrides.yaml', 'w') as f:
            data = omegaconf.OmegaConf.to_container(handler_config)
            yaml.dump({'handlers': data}, f, default_style=None, default_flow_style=False)
        return True
    return False

def archive(cfg):
    hydra.utils.log.debug(f"Configuration:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")    
    omegaconf.omegaconf.OmegaConf.set_struct(cfg, False)
    args = []
    args += ["torch-model-archiver"]
    args += ["--model-name", cfg.archive.name]
    args += ["--version", str(cfg.archive.version)]
    args += ["--handler", optimizer_server.__file__ if cfg.archive.mode == 'fit' else model_server.__file__]
    args += ["--serialized-file", toolz.get_in(['archive', 'ckpt'], cfg) or '']
    yaml_files = get_files(cfg.archive.root, cfg.archive.conf, "*.yaml")
    py_files = get_files(cfg.archive.root, cfg.archive.src, "*.py")
    dump_zipfile("conf.zip", cfg.archive.root, yaml_files)
    dump_zipfile("src.zip", cfg.archive.root, py_files)    
    extra_files = "conf.zip,src.zip,.hydra/overrides.yaml,pre.yaml,post.yaml"
    if dump_handlers(cfg.archive.handlers, cfg.handlers):
        extra_files += ",handler_overrides.yaml"
    args += [f"--extra-files", extra_files]
    # confs = [cfg.archive.conf] if isinstance(cfg.archive.conf, str) else cfg.archive.conf
    # srcs = [cfg.archive.src] if isinstance(cfg.archive.src, str) else cfg.archive.src
    # args += [f"--extra-files", ",".join(map(lambda p: os.path.join(cfg.archive.root, p, ''), confs + srcs))]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=None, shell=True)
    output = proc.communicate()[0].decode('UTF-8')
    if not output:
        log.info(f"Model successfully archived @ {os.getcwd()}")
    else:
        log.error(f"An error has occured while archiving the model:\n{output}")
    
if __name__ == "__main__":  
    config_filename = sys.argv.pop(1) #TODO: argparser integration?
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${archive.name}")
    archive = hydra.main(config_path="conf", config_name=config_filename)(archive)
    archive()