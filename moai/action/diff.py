import os
import omegaconf as oconf
import deepdiff as dd
import hydra
import logging
import itertools
import rich
from rich.console import Console
from rich import box

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def get_conf(path: str) -> oconf.OmegaConf:
    if os.path.splitext(path)[1] != '.yaml':
        nominal_config_path = os.path.join(path, 'config_resolved.yaml')
        if os.path.exists(nominal_config_path):
            path = nominal_config_path
        elif os.path.basename(path) != '.hydra':
            path = os.path.join(path, '.hydra') #TODO: merge with overrides
            path = os.path.join(path, 'config.yaml')
    return oconf.OmegaConf.load(path)

def report_hydra(ddiff: dd.DeepDiff) -> None:
    def add_item(txt: str) -> None:
        log.debug(f"\t{txt}")
    def remove_item(txt: str) -> None:
        log.error(f"\t{txt}")
    def value_changed(txt: str) -> None:
        log.warning(f"\t{txt}")
    def key_changed(txt: str) -> None:
        log.critical(f"\t{txt}")

    def format_added_item(txt: str) -> str:
        ret = txt.replace('\'', '').\
            replace('root', '').\
            replace('[', '').\
            replace(']', '.')
        return ret[:-1] if ret.endswith('.') else ret

    def format_value_changed(txt: str) -> str:
        old = txt[1]['old_value']
        new = txt[1]['new_value']
        return txt[0].replace('\'','').\
            replace('][', '.').\
            replace('root[', '').\
            replace(']','')\
            + f': {old} -> {new}'
 
    def format_iterable_item(txt: str) -> str:
        ret = txt[:-3].replace('\'', '').\
            replace('root', '').\
            replace('[', '').\
            replace(']', '.') 
        return (ret[:-1] if ret.endswith('.') else ret) + txt[-3:]

    if 'dictionary_item_removed' in ddiff.keys():
        log.info("- Dictionary Items Removed:")
        for i in ddiff['dictionary_item_removed'].items:
            formatted = format_added_item(i)
            remove_item(formatted)

    if 'dictionary_item_added' in ddiff.keys():
        log.info("+ Dictionary Items Added:")
        for i in ddiff['dictionary_item_added'].items:
          formatted = format_added_item(i)
          add_item(formatted)

    if 'iterable_item_added' in ddiff.keys():
        log.info("+ Iterable Items Added:")
        for i in ddiff['iterable_item_added'].items():
          add_item(f'{format_iterable_item(i[0])} -> {format_iterable_item(i[1])}')
    
    if 'iterable_item_removed' in ddiff.keys():
        log.info("- Iterable Items Removed:")
        for i in ddiff['iterable_item_removed'].items():
          remove_item(f'{format_iterable_item(i[0])} -> {format_iterable_item(i[1])}')
    
    if 'type_changes' in ddiff.keys():
        log.info("~ Types Changed:")
        for i in ddiff['type_changes'].items():
          key_changed(i)

    if 'values_changed' in ddiff.keys():
        log.info("~ Values Changed:")
        for i in ddiff['values_changed'].items():
          value_changed(format_value_changed(i))

    if 'repetition_change' in ddiff.keys():
        log.info("~ Repetition Changed:")
        for i in ddiff['repetition_change'].items():
          value_changed(i)

    if 'set_item_added' in ddiff.keys():    
        log.info("+ Set Item Added:")
        for i in ddiff['set_item_added'].items():
          add_item(i)

    if 'set_item_removed' in ddiff.keys():    
        log.info("- Set Item Removed:")
        for i in ddiff['set_item_removed'].items():
          remove_item(i)

def report_rich(ddiff: dd.DeepDiff, console: Console) -> None:
    key_table = rich.table.Table(box=box.SIMPLE_HEAVY, border_style="bold yellow")
    key_table.add_column(":heavy_check_mark: [bold green] Added", justify='center')
    key_table.add_column(":x: [bold red] Removed", justify='center')

    collection_table = rich.table.Table(box=box.SIMPLE_HEAVY, border_style="bold blue")
    collection_table.add_column(":key: [bold blue] Key", justify='center')
    collection_table.add_column(":x: [bold red] From", justify='center')
    collection_table.add_column(":heavy_check_mark: [bold green] To", justify='center')

    value_table = rich.table.Table(box=box.SIMPLE_HEAVY, border_style='bold cyan')
    value_table.add_column(":key: [bold cyan] Key", justify='center')
    value_table.add_column(":x: [bold red] From", justify='center')
    value_table.add_column(":heavy_check_mark: [bold green] To", justify='center')

    results = {
        'item': {
            'added': [], 'removed': [],
        },
        'collection': {
            'added': {}, 'removed': {},
        },
        'changed': {
            'new': [], 'key': [], 'old': [], 
        },
        'repetition': {
            'added': [], 'removed': [],
        }
    }

    def format_added_item(txt: str) -> str:
        ret = txt.replace('\'', '').\
            replace('root', '').\
            replace('[', '').\
            replace(']', '.')
        return ret[:-1] if ret.endswith('.') else ret

    def format_value_changed(txt: str) -> str:
        old = txt[1]['old_value']
        new = txt[1]['new_value']
        return txt[0].replace('\'','').\
            replace('][', '.').\
            replace('root[', '').\
            replace(']','')\
            + f': {old} -> {new}'
 
    def format_iterable_item(txt: str) -> str:
        ret = txt[:-3].replace('\'', '').\
            replace('root', '').\
            replace('[', '').\
            replace(']', '.') 
        return (ret[:-1] if ret.endswith('.') else ret) + txt[-3:]

    if 'dictionary_item_removed' in ddiff.keys():
        for i in ddiff['dictionary_item_removed'].items:
            formatted = format_added_item(i)
            results['item']['removed'].append(formatted)

    if 'dictionary_item_added' in ddiff.keys():
        for i in ddiff['dictionary_item_added'].items:
          formatted = format_added_item(i)
          results['item']['added'].append(formatted)

    if 'iterable_item_added' in ddiff.keys():
        for i in ddiff['iterable_item_added'].items():
          key = format_iterable_item(i[0])
          value = format_iterable_item(i[1] if isinstance(i[1], str) else str(i[1]))
          results['collection']['added'][key] = value
    
    if 'iterable_item_removed' in ddiff.keys():
        for i in ddiff['iterable_item_removed'].items():
          key = format_iterable_item(i[0])
          value = format_iterable_item(i[1] if isinstance(i[1], str) else str(i[1]))
          results['collection']['removed'][key] = value
    
    if 'type_changes' in ddiff.keys():
        for i in ddiff['type_changes'].items():
            if isinstance(i, tuple):                
                results['changed']['key'].append(format_added_item(i[0]))
                results['changed']['new'].append(i[1]['new_value'])
                results['changed']['old'].append(i[1]['old_value'])
            else:
                results['changed']['key'].append(i)

    if 'values_changed' in ddiff.keys():
        for i in ddiff['values_changed'].items():            
            results['changed']['key'].append(format_added_item(i[0]))
            results['changed']['new'].append(i[1]['new_value'])
            results['changed']['old'].append(i[1]['old_value'])

    if 'repetition_change' in ddiff.keys():
        for i in ddiff['repetition_change'].items():
            pass

    if 'set_item_added' in ddiff.keys():    
        for i in ddiff['set_item_added'].items():
            pass

    if 'set_item_removed' in ddiff.keys():    
        for i in ddiff['set_item_removed'].items():
            pass

    if results['item']['added']:
        console.rule("Keys Diffs", style="bold yellow")
        for i, (a, r) in enumerate(itertools.zip_longest(
            results['item']['added'], 
            results['item']['removed'])
        ):
            key_table.add_row(
                f'[bold green] {a}' if a is not None else '',
                f'[bold red] {r}' if r is not None else '',
                end_section=(i == len(results['item']['added']) - 1)
            )
        console.print(key_table, justify='center')

    collection_added = set(results['collection']['added'].keys())
    collection_removed = set(results['collection']['removed'].keys())
    collection_diffs = collection_added.intersection(collection_removed)
    collection_removed = collection_removed - collection_diffs
    collection_added = collection_added - collection_diffs
    if collection_diffs:
        console.rule("Collections Diffs", style="bold blue")
        for k in collection_diffs:
            a = results['collection']['added'][k]
            r = results['collection']['removed'][k]
            collection_table.add_row(
                f'[bold blue] {k}',
                f'[bold red] {r}',
                f'[bold green] {a}'
            )
    if collection_added:
        for k in collection_added:
            a = results['collection']['added'][k]            
            collection_table.add_row(
                f'[bold blue] {k}',
                '',
                f'[bold green] {a}'
            )
    if collection_removed:
        for k in collection_removed:
            r = results['collection']['removed'][k]            
            collection_table.add_row(
                f'[bold blue] {k}',
                f'[bold red] {r}'
                '',                
            )
    if collection_added or collection_removed or collection_diffs:
        console.print(collection_table, justify='center')
    
    if results['changed']['key']:
        console.rule("[bold syan] Value Diffs", style="bold cyan")
        for k, n, o in zip(        
            results['changed']['key'],
            results['changed']['new'], 
            results['changed']['old'], 
        ):
            value_table.add_row(
                f'[bold cyan] {k}',
                f'[bold red] {o}',
                f'[bold green] {n}',                
            )
        console.print(value_table, justify='center')
        
def replace_bool(conf: dict) -> None:
    for k, v in conf.items():
        if isinstance(v, bool):
            conf[k] = str(v)
        if isinstance(v, oconf.ListConfig):
            for i, b in enumerate(v):
                if isinstance(b, bool):
                    conf[k][i] = str(b)
        if isinstance(v, oconf.DictConfig):
            replace_bool(v)

#NOTE: https://github.com/ikatyang/emoji-cheat-sheet 

@hydra.main(config_name="tools/diff.yaml", config_path="conf", version_base='1.3')
def diff(cfg):    
    console = Console()
    console.rule("Config Diff", style="magenta")
    from_style = rich.style.Style(frame=True)
    console.line()
    console.print("[bold red]From:[/bold red] :arrow_right_hook:", cfg.lhs, style=from_style)    
    # console.rule()
    to_style = rich.style.Style(frame=True)
    console.print("[bold green]To:[/bold green] ", cfg.rhs, justify='right', style=to_style)
    console.line()
    # console.rule()
    with console.status("Loading configs ...", spinner='aesthetic') as status:
        lhs = get_conf(cfg.lhs)
        rhs = get_conf(cfg.rhs)
        replace_bool(lhs) # bool is unhashable
        replace_bool(rhs) # bool is unhashable
    with console.status("Calculating diffs ...", spinner='aesthetic') as status:
        ddiff = dd.DeepDiff(lhs, rhs, ignore_order=True)
    # report_hydra(ddiff)
    report_rich(ddiff, console)


if __name__ == "__main__":
    # os.environ['HYDRA_FULL_ERROR'] = str(1)
    diff()
