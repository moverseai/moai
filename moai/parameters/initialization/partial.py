import logging
import os
import re
from collections import OrderedDict

import omegaconf.omegaconf
import toolz
import torch

from moai.utils.funcs import select, select_list
from moai.utils.torch import get_submodule

log = logging.getLogger(__name__)

__all__ = ["Partial"]

"""
    modules.module1:
        from: &ckpt_path PATH/TO/CKPT
        keys: [*] # glob pattern, * means all
        # strict: true # default is true
        replace: [] or null # default dont replace anything
    modules.module2.encoder:
        from: *ckpt_path
        keys: [encoder.*]
        strict: false
        replace:
            - source: encoder.
                target: ''
    monads.proc1:
        from: PATH/TO/CKPT2
        keys: [*]
        replace: []
"""

__escaped_glob_tokens_to_re__ = OrderedDict(
    (
        # Order of ``**/`` and ``/**`` in RE tokenization pattern doesn't matter because ``**/`` will be caught first no matter what, making ``/**`` the only option later on.
        # W/o leading or trailing ``/`` two consecutive asterisks will be treated as literals.
        (
            "/\*\*",
            "(?:/.+?)*",
        ),  # Edge-case #1. Catches recursive globs in the middle of path. Requires edge case #2 handled after this case.
        (
            "\*\*/",
            "(?:^.+?/)*",
        ),  # Edge-case #2. Catches recursive globs at the start of path. Requires edge case #1 handled before this case. ``^`` is used to ensure proper location for ``**/``.
        (
            "\*",
            "[^/]*",
        ),  # ``[^/]*`` is used to ensure that ``*`` won't match subdirs, as with naive ``.*?`` solution.
        ("\?", "."),
        ("\[\*\]", "\*"),  # Escaped special glob character.
        ("\[\?\]", "\?"),  # Escaped special glob character.
        (
            "\[!",
            "[^",
        ),  # Requires ordered dict, so that ``\[!`` preceded ``\[`` in RE pattern. Needed mostly to differentiate between ``!`` used within character class ``[]`` and outside of it, to avoid faulty conversion.
        ("\[", "["),
        ("\]", "]"),
    )
)

__escaped_glob_replacement__ = re.compile(
    "(%s)" % "|".join(__escaped_glob_tokens_to_re__).replace("\\", "\\\\\\")
)


def __glob_to_re__(pattern):
    return __escaped_glob_replacement__.sub(
        lambda match: __escaped_glob_tokens_to_re__[match.group(0)], re.escape(pattern)
    )


class Partial(object):
    def __init__(
        self,
        **kwargs: omegaconf.DictConfig,
    ):
        self.kwargs = kwargs

    def __call__(self, module: torch.nn.Module) -> None:
        for component_name, init_config in self.kwargs.items():
            if (m := get_submodule(module, component_name)) is not None:
                if (ckpt := select(init_config, "from")) is None or not os.path.exists(
                    ckpt
                ):
                    log.error(
                        f"Checkpoint file `[ul cyan]{ckpt}[/]` does not exist :exclamation:"
                    )
                if strict := select(init_config, "strict") is None:
                    strict = True
                log.info(
                    f"Initializing [italic cyan]{component_name}[/] "
                    + (
                        "[bold red]\[strictly][/]"
                        if strict
                        else "[bold yellow]\[relaxed][/]"
                    )
                    + f" from [ul]{ckpt}[/]."
                )
                data = torch.load(ckpt, map_location=lambda s, l: s)
                state_dict = data["state_dict"]
                selected_keys = []
                for key in select_list(init_config, "keys"):
                    pattern = __glob_to_re__(key)
                    matches = [re.fullmatch(pattern, k) for k in state_dict.keys()]
                    matches = [m for m in matches if bool(m)]
                    selected_keys.extend(str(m.string) for m in matches)
                state_dict = toolz.keyfilter(lambda k: k in selected_keys, state_dict)
                replace_config = select_list(init_config, "replace")
                # NOTE: always add the current components name to be replaced as we init the same module
                replace_config.insert(0, {"source": f"{component_name}."})
                for replace in replace_config:
                    target = select(replace, "target") or ""
                    state_dict = toolz.keymap(
                        lambda txt: txt.replace(replace.source, target), state_dict
                    )
                m.load_state_dict(state_dict, strict=strict)
            else:
                log.warning(
                    f"[bold yellow]:warning:[/] Could not locate [red bold]{component_name}[/] when attempting to partially initialize it."
                )
