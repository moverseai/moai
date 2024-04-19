from moai.utils.arguments import ensure_numeric_list

import random
import sys
import os
import hydra
import omegaconf.omegaconf
import hiplot as hip
import logging
import typing
import toolz
import yaml
import datetime
import parsedatetime as pdt
import glob
import pandas as pd
import tqdm
import functools

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

__DATE_TIME_FORMAT__ = "%Y-%m-%d-%H-%M-%S"

__DATE_TIME_CONVERTERS__ = {
    1: lambda d: datetime.datetime.combine(d, datetime.time()), # date
    2: lambda t: datetime.datetime.combine(datetime.date(), t), # time
    3: lambda dt: dt, # datetime
}

def parse_machine_date_time(string: str) -> datetime.datetime: 
    return datetime.datetime.strptime(string, __DATE_TIME_FORMAT__)

def parse_human_date_time(day: str, time: str) -> datetime.datetime:
    cal = pdt.Calendar()
    d = day if day else ''
    t = time if time else ''
    dt, result = cal.parseDT(f"{d} {t}".strip())
    if not result:
        log.error(f"Given day/time arguments ({day}/{time}) are not valid.")    
    return __DATE_TIME_CONVERTERS__[result](dt)

def get_conf(path: str) -> omegaconf.OmegaConf:
    if os.path.splitext(path)[1] != '.yaml':
        nominal_config_path = os.path.join(path, 'config_resolved.yaml')
        if os.path.exists(nominal_config_path):
            path = nominal_config_path
        elif os.path.basename(path) != '.hydra':
            path = os.path.join(path, '.hydra') #TODO: merge with overrides
            path = os.path.join(path, 'config.yaml')
    return path

def safe_get_dict(key: str, dictionary: dict):
    return toolz.get(key, dictionary, { })

def extract_losses(
    config:             omegaconf.DictConfig,
    path:               str,
): 
    result = { }
    supervision = toolz.get_in(["model", "supervision"], config)
    supervision = toolz.dissoc(supervision, "losses", "_target_")        
    for k, v in supervision.items():
        w = v['weight'] if 'weight' in v else 1.0
        w = sum(w) if isinstance(w, list) else w
        result[k] = w
    return result

def extract_metrics(
    moai_config:            omegaconf.DictConfig,
    path:                   str,
    options:                typing.Union[str, int],
):
    val_csv = glob.glob(os.path.join(path, "*_val.csv"))
    result = { }
    if val_csv:
        result = { }
        metrics = pd.read_csv(toolz.first(val_csv))
        metric_names = toolz.get_in(["model", "validation"], moai_config)
        metric_names = toolz.dissoc(metric_names, "metrics", "_target_", "indicators")
        if options.epoch_mode == 'best':
            best_metrics = { }
            for k, v in metric_names.items():
                name = v['out'] if 'out' in v else k
                name = os.path.commonprefix(name) if isinstance(name, list) else name
                best_metrics[name] = (metrics[name].max(), metrics['epoch'][metrics[name].idxmax()])
            min_epoch = int(min(map(lambda v: v[1], best_metrics.values())))        
            result['epoch'] = min_epoch
            for k, (v, e) in best_metrics.items():
                result[k] = v
                if e == min_epoch:
                    result['best'] = k
        else:
            epoch_value = [0, sys.maxsize] if options.epoch_mode == 'all' else ensure_numeric_list(options.epoch_value)
            min_epoch, max_epoch = tuple(epoch_value) if isinstance(epoch_value, omegaconf.ListConfig) and len(epoch_value) == 2 else (epoch_value[0], epoch_value[0])
            for i, e in enumerate(metrics['epoch']):
                e = int(e)
                if e >= min_epoch and e <= max_epoch:
                    result[i] = { }
                    result[i]['epoch'] = e
            for k, v in metric_names.items():
                name = v['out'] if 'out' in v else k
                name = os.path.commonprefix(name) if isinstance(name, list) else name
                for i, (e, v) in enumerate(zip(metrics['epoch'], metrics[name])):
                    e = int(e)
                    if e >= min_epoch and e <= max_epoch:
                        result[i][name] = v

    else:
        log.error(f"Could not find a validation metrics .csv file in {path}, skipping metrics for this experiment.")
    return result

def extract_optimization(
    moai_config:             omegaconf.DictConfig,
):
    result = { }
    optimizer = toolz.get_in(["model", "parameters", "optimization"], moai_config)
    learning_rates = { }
    lr = toolz.get("lr", optimizer, default=None)
    if lr:
        learning_rates[optimizer['_target_'].split('.')[-1]] = lr
    optimizers = toolz.get("optimizers", optimizer, default=None)
    if optimizers and isinstance(optimizers, dict):
        if len(optimizers) == 1 and "lr" in toolz.first(optimizers.values()):
            learning_rates[toolz.first(optimizers.keys())] = toolz.first(optimizers.values())["lr"]
        else:
            for optim, params in optimizers.items():
                if toolz.get("lr", params, default=None):
                    learning_rates[optim] = params["lr"]
    for k, v in learning_rates.items():
        result['optimizer'] = k
        result['lr'] = v
    return result

def extract_data(
    path: str, 
    plot_config: omegaconf.DictConfig
) -> typing.Mapping[str, typing.Union[str, float]]:
    data_frame = { }
    config_file = get_conf(path)
    with open(config_file, 'r') as f:
        moai_config = yaml.load(f, Loader=yaml.SafeLoader)
    if plot_config.losses:
        data_frame['losses'] = extract_losses(moai_config, path)
    if plot_config.metrics:
        data_frame['metrics'] = extract_metrics(moai_config, path, plot_config.metrics_options)
    if plot_config.optimizer: #TODO: add params & init func (i.e. pretrained)
        data_frame['optimization'] = extract_optimization(moai_config)
    if plot_config.monads:
        monads = toolz.get_in(["model", "monads"], moai_config)
        if plot_config.monads_options.key:
            data_frame['monads'] = { }
            for key in plot_config.monads_options.key:
                data_frame['monads'][key] = 'yes' if key in monads else 'no'
        if plot_config.monads_options.value:
            for k, v in plot_config.monads_options.value.items():
                data_frame['monads'][f"{k}_{v}"] = toolz.get_in([k, v], monads)
    #TODO: add model related data
    return data_frame

def post_process(
    data: typing.List[typing.Dict[str, typing.Dict[str, typing.Union[str, float, int]]]],
    config: omegaconf.DictConfig,
) -> typing.List[typing.Dict[str, typing.Union[str, float, int]]]:
    if config.losses:
        log.info("Adding losses ...")
        loss_keys = functools.reduce(lambda s, d: s | set(safe_get_dict('losses', d).keys()), data, set())
        losses_empty = dict(((k, 0.0) for k in loss_keys))
    if config.metrics:
        log.info("Adding metrics ...")
        metric_keys = functools.reduce(lambda s, d: s | set(safe_get_dict('metrics', d).keys()), data, set())
        metrics_empty = dict(((k, 0.0) for k in metric_keys))
    if config.monads:
        log.info("Adding monads ...")
        monads_keys = functools.reduce(lambda s, d: s | set(safe_get_dict('monads', d).keys()), data, set())
        monads_empty = dict(((k, 0.0) for k in monads_keys))
    consolidated_data = []
    for datum in data:
        res = { 'date': datum['date'] }
        if config.optimizer:
            res = toolz.merge(res, toolz.get('optimization', datum, { }))
        if config.losses:
            res = toolz.merge(res, losses_empty, toolz.get('losses', datum, { }))
        if config.monads:
            res = toolz.merge(res, monads_empty, toolz.get('monads', datum, { }))
        if config.metrics:
            if config.metrics_options.epoch_mode == 'best':
                res = toolz.merge(res, metrics_empty, toolz.get('metrics', datum, { }))
                consolidated_data.append(res)
            else:
                for row in datum['metrics'].values():
                    consolidated_data.append(toolz.merge(res, row))
    return consolidated_data

@hydra.main(config_name="tools/plot.yaml", config_path="conf", version_base='1.2')
def plot(cfg):
    from_date = parse_human_date_time(cfg.filter.date.start.day, cfg.filter.date.start.time)
    to_date = parse_human_date_time(cfg.filter.date.end.day, cfg.filter.date.end.time)
    if from_date > to_date:
        log.error("Invalid date ranges, start date ({from_date}) should be earlier than end date ({to_date}) date.")
        return
    log.info(f"Parsing {cfg.filter.name} experiments from {from_date} to {to_date}.")
    root = cfg.root
    date_paths = []
    root = root if os.path.isabs(root) else os.path.join(
        hydra.utils.get_original_cwd(), root
    )
    for date_folder in filter(
        lambda d: os.path.isdir(os.path.join(root, d)),
        os.listdir(root)
    ):
        for day_folder in filter(
            lambda d: os.path.isdir(os.path.join(root, date_folder, d)),
            os.listdir(os.path.join(root, date_folder))
        ):
            combined = f"{date_folder}-{day_folder}"[:19]
            date = parse_machine_date_time(combined)
            date_paths.append((combined, date, os.path.join(root, date_folder, day_folder)))
    paths = [(datetime, folder) for (datetime, date, folder) in date_paths if from_date <= date <= to_date]
    if not len(paths):
        log.error("Found no experiments")
        return
    log.info(f"Found {len(paths)} experiments.")
    data = []
    for path in tqdm.tqdm(paths, desc="Extracting data"):
        datum = extract_data(path[1], cfg)
        datum['date'] = path[0]
        data.append(datum)
    data = post_process(data, cfg)
    if data:
        hip.Experiment.from_iterable(data).to_html("plot.html")
        os.system("plot.html")
    else:
        log.error("No data collected.")

if __name__ == "__main__":
    # os.environ['HYDRA_FULL_ERROR'] = '1'
    sys.argv.append("hydra.run.dir=actions/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    plot()
