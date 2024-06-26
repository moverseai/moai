import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import typing
import zipfile

import hydra
import omegaconf.omegaconf
import toolz
import yaml

log = logging.getLogger(__name__)

try:
    import moai.serve.model as model_server
    import moai.serve.optimizer as optimizer_server
    import moai.serve.streaming_optimizer as streaming_optimizer_server
except ImportError:
    log.warning(
        "Archive action is unavailable, please make sure you have installed all the dependencies (e.g. ts, model-archiver)."
    )


def get_files(
    root: str,
    folder: typing.Union[str, typing.Sequence[str]],
    pattern: str,
) -> typing.Sequence[str]:
    folders = [folder] if isinstance(folder, str) else folder
    folders = [os.path.join(root, f) for f in folders]
    files = []
    for f in folders:
        path = (
            f if os.path.isabs(f) else os.path.join(hydra.utils.get_original_cwd(), f)
        )
        files += glob.glob(os.path.join(path, pattern))
        files += glob.glob(os.path.join(path, "**", pattern), recursive=True)
    return list(toolz.unique(files))


def dump_zipfile(
    filename: str,
    root: str,
    files: typing.Sequence[str],
):
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as z:
        for file in files:
            arcname = (
                os.path.join(
                    "." if "nt" in os.name else os.getcwd(), file.replace(root, "")
                )
                if "model.py" not in file
                else "./model.py"
            )
            log.info(f"Adding file ({file}) with name: {arcname}")
            z.write(file, arcname)


def dump_handlers(
    handlers: omegaconf.omegaconf.DictConfig,
    handler_config: omegaconf.omegaconf.DictConfig,
    streaming_handler_config: omegaconf.omegaconf.DictConfig = None,
):
    empty = omegaconf.DictConfig({})
    # parse nested lists
    # Use regular expression to extract elements within square brackets
    pattern = r"\[(.*?)\]"
    with open("pre.yaml", "w") as f:
        data = omegaconf.OmegaConf.to_container(handlers.get("pre") or empty)
        # data = list(
        #     map(lambda e: {e.split(":")[0].strip(): e.split(":")[1].strip()}, data)
        # )
        new_data = []
        for e in data:
            key, value = e.split(":")
            match = re.search(pattern, value)
            if match:
                # Extract elements within square brackets
                data_list = match.group(1).split(",")
                # Remove leading/trailing spaces from elements
                data_list = [item.strip() for item in data_list]
                new_data.append({key.strip(): data_list})
            else:
                new_data.append({key.strip(): value.strip()})
        yaml.dump(
            {"defaults": new_data}, f, default_style=None, default_flow_style=False
        )
    with open("post.yaml", "w") as f:
        data = omegaconf.OmegaConf.to_container(handlers.get("post") or empty)
        # data = list(
        # map(lambda e: {e.split(":")[0].strip(): e.split(":")[1].strip()}, data)
        # )
        new_data = []
        for e in data:
            key, value = e.split(":")
            match = re.search(pattern, value)
            if match:
                # Extract elements within square brackets
                data_list = match.group(1).split(",")
                # Remove leading/trailing spaces from elements
                data_list = [item.strip() for item in data_list]
                new_data.append({key.strip(): data_list})
            else:
                new_data.append({key.strip(): value.strip()})
        yaml.dump(
            {"defaults": new_data}, f, default_style=None, default_flow_style=False
        )
    if handler_config is not None:
        with open("handler_overrides.yaml", "w") as f:
            data = omegaconf.OmegaConf.to_container(handler_config)
            yaml.dump(
                {"handlers": data}, f, default_style=None, default_flow_style=False
            )
    if streaming_handler_config is not None:
        with open("streaming_handler_overrides.yaml", "w") as f:
            data = omegaconf.OmegaConf.to_container(streaming_handler_config)
            yaml.dump(
                {"streaming_handlers": data},
                f,
                default_style=None,
                default_flow_style=False,
            )
    return True


def archive(cfg):
    hydra.utils.log.debug(
        f"Configuration:\n{omegaconf.OmegaConf.to_container(cfg, resolve=True)}"
    )
    omegaconf.omegaconf.OmegaConf.set_struct(cfg, False)
    args = []
    args += ["torch-model-archiver"]
    args += ["--model-name", cfg.archive.name]
    args += ["--version", str(cfg.archive.version)]
    extra_files = "conf.zip,src.zip,.hydra/overrides.yaml,pre.yaml,post.yaml"
    py_files = get_files(cfg.archive.root, cfg.archive.src, "*.py")
    if omegaconf.OmegaConf.select(cfg.archive, "force"):
        args += ["--force"]
    if cfg.archive.mode == "fit":
        py_files.append(model_server.__file__)
        args += ["--handler", optimizer_server.__file__]
    elif cfg.archive.mode == "streaming":
        py_files.append(model_server.__file__)
        # TODO: add model ovverides to streaming
        # would be the keys of tensors to keep in memory (e.g. embeddings)
        args += ["--handler", streaming_optimizer_server.__file__]
    else:
        args += ["--handler", model_server.__file__]
    args += ["--serialized-file", toolz.get_in(["archive", "ckpt"], cfg) or ""]
    yaml_files = get_files(cfg.archive.root, cfg.archive.conf, "*.yaml")
    dump_zipfile("conf.zip", cfg.archive.root, yaml_files)
    dump_zipfile("src.zip", cfg.archive.root, py_files)
    if dump_handlers(
        omegaconf.OmegaConf.select(cfg.archive, "handlers"),
        omegaconf.OmegaConf.select(cfg, "handlers"),
        omegaconf.OmegaConf.select(cfg, "streaming_handlers"),
    ):
        extra_files += (
            ",handler_overrides.yaml"
            if omegaconf.OmegaConf.select(cfg, "streaming_handlers") is None
            else ",handler_overrides.yaml,streaming_handler_overrides.yaml"
        )
    args += [f"--extra-files", extra_files]
    if omegaconf.OmegaConf.select(cfg.archive, "requirements"):
        requirements = (
            cfg.archive.requirements
            if os.path.exists(cfg.archive.requirements)
            else os.path.join(cfg.archive.root, cfg.archive.requirements)
        )
        # NOTE: try hydra original cwd as well?
        if not os.path.exists(requirements):
            log.warning(
                f"Requirements file ({requirements}) not found, skipping its packaging."
            )
        else:
            basename = os.path.basename(requirements)
            shutil.copyfile(requirements, os.path.join(os.getcwd(), basename))
            args += ["-r", basename]
    if omegaconf.OmegaConf.select(cfg.archive, "output_path"):  # NOTE
        if not os.path.exists(cfg.archive.output_path):
            os.makedirs(cfg.archive.output_path, exist_ok=True)
        args += ["--export-path", cfg.archive.output_path]
    log.info(f"Running: {args}")
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=None)  # , shell=True)
    output = proc.communicate()[0].decode("UTF-8")
    if not output:
        log.info(
            f"Model successfully archived @ {cfg.archive.output_path if cfg.archive.output_path  else os.getcwd()}"
        )
    else:
        log.error(f"An error has occured while archiving the model:\n{output}")


if __name__ == "__main__":
    config_filename = sys.argv.pop(1)  # TODO: argparser integration?
    sys.argv.append(
        "hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}-${archive.name}"
    )
    archive = hydra.main(
        config_path="conf", config_name=config_filename, version_base="1.3"
    )(archive)
    archive()
