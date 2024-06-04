import inspect
import logging
import typing
from collections import defaultdict

import hydra.utils as hyu
import numpy as np
import omegaconf.omegaconf
import toolz
import torch

import moai.networks.lightning as minet
import moai.nn.linear as mil
import moai.utils.parsing.rtp as mirtp
from moai.networks.lightning.feedforward import (
    _create_processing_block,
    _create_validation_block,
)

log = logging.getLogger(__name__)

__all__ = ["VariationalAutoencoder"]

# NotImplementedError:
# Support for `validation_epoch_end` has been removed in v2.0.0.
# `VariationalAutoencoder` implements this method.
# You can use the `on_validation_epoch_end` hook instead.
# To access outputs, save them in-memory as instance attributes.
# You can find migration examples in https://github.com/Lightning-AI/lightning/pull/16520.


class VariationalAutoencoder(minet.FeedForward):
    def __init__(
        self,
        configuration: omegaconf.DictConfig,
        io: omegaconf.DictConfig,
        modules: omegaconf.DictConfig,
        data: omegaconf.DictConfig = None,
        parameters: omegaconf.DictConfig = None,
        feedforward: omegaconf.DictConfig = None,
        monads: omegaconf.DictConfig = None,
        supervision: omegaconf.DictConfig = None,
        validation: omegaconf.DictConfig = None,
        generation: omegaconf.DictConfig = None,
        visualization: omegaconf.DictConfig = None,
        export: omegaconf.DictConfig = None,
    ):
        super(VariationalAutoencoder, self).__init__(
            feedforward=feedforward,
            monads=monads,
            visualization=visualization,
            data=data,
            supervision=supervision,
            validation=validation,
            export=export,
            parameters=parameters,
        )
        # self.latent_dim = configuration.features_head.latent_dim
        self.repeat_val = configuration.repeat_val
        self.traversal_len = configuration.traversal_len
        self.traversal_step = configuration.traversal_step
        self.traversal_dim = configuration.traversal_dim
        self.traversal_init_value = configuration.traversal_init_value
        self.encoder = hyu.instantiate(modules["encoder"])
        # responsible for predicting the latent distribution (e.g. mu, logvar, etc.)
        self.latent_dist_predictor = hyu.instantiate(modules["latent_dist_predictor"])
        self.sampler = hyu.instantiate(modules["sampler"])
        self.decoder = hyu.instantiate(modules["decoder"])
        self.sample = _create_processing_block(generation, "sample", monads=monads)
        self.walk = _create_processing_block(generation, "walk", monads=monads)
        self.generate = _create_processing_block(generation, "generate", monads=monads)
        self.gen_validation = _create_validation_block(
            toolz.get("generation", validation, None)
        )
        (
            self.enc_fwds,
            self.dec_fwds,
            self.f2mu_std_fwds,
            self.rep_fwds,
            self.gen_fwds,
        ) = ([], [], [], [], [])

        params = inspect.signature(self.encoder.forward).parameters
        enc_in = list(zip(*[mirtp.force_list(io.encoder[prop]) for prop in params]))
        enc_out = mirtp.split_as(mirtp.resolve_io_config(io.encoder["out"]), enc_in)

        self.enc_res_fill = [
            mirtp.get_result_fillers(self.encoder, out) for out in enc_out
        ]
        get_enc_filler = iter(self.enc_res_fill)

        for keys in enc_in:
            self.enc_fwds.append(
                lambda td, tk=keys, args=params.keys(), enc=self.encoder, filler=next(
                    get_enc_filler
                ): filler(
                    td,
                    enc(
                        **dict(
                            zip(
                                args,
                                list(
                                    td[k] if type(k) is str else list(td[j] for j in k)
                                    for k in tk
                                ),
                            )
                        )
                    ),
                )
            )

        if self.latent_dist_predictor is not None:
            params = inspect.signature(self.latent_dist_predictor.forward).parameters
            ldist_in = list(
                zip(
                    *[
                        mirtp.force_list(io.latent_dist_predictor[prop])
                        for prop in params
                    ]
                )
            )
            ldist_out = mirtp.split_as(
                mirtp.resolve_io_config(io.latent_dist_predictor["out"]), ldist_in
            )

            self.ldist_res_fill = [
                mirtp.get_result_fillers(self.latent_dist_predictor, out)
                for out in ldist_out
            ]
            get_ldist_filler = iter(self.ldist_res_fill)

            for keys in ldist_in:
                self.f2mu_std_fwds.append(
                    lambda td, tk=keys, args=params.keys(), rep=self.latent_dist_predictor, filler=next(
                        get_ldist_filler
                    ): filler(
                        td,
                        rep(
                            **dict(
                                zip(
                                    args,
                                    list(
                                        (
                                            td[k]
                                            if type(k) is str
                                            else list(td[j] for j in k)
                                        )
                                        for k in tk
                                    ),
                                )
                            )
                        ),
                    )
                )
        else:
            log.warning(
                "No latent distribution predictor found. Assuming standard normal distribution."
            )

        if self.sampler is not None:
            params = inspect.signature(self.sampler.forward).parameters
            sampler_in = list(
                zip(*[mirtp.force_list(io.sampler[prop]) for prop in params])
            )
            sampler_out = mirtp.split_as(
                mirtp.resolve_io_config(io.sampler["out"]), sampler_in
            )

            self.sampler_res_fill = [
                mirtp.get_result_fillers(self.sampler, out) for out in sampler_out
            ]
            get_sampler_filler = iter(self.sampler_res_fill)

            for keys in sampler_in:
                self.rep_fwds.append(
                    lambda td, tk=keys, args=params.keys(), rep=self.sampler, filler=next(
                        get_sampler_filler
                    ): filler(
                        td,
                        rep(
                            **dict(
                                zip(
                                    args,
                                    list(
                                        (
                                            td[k]
                                            if type(k) is str
                                            else list(td[j] for j in k)
                                        )
                                        for k in tk
                                    ),
                                )
                            )
                        ),
                    )
                )
        else:
            log.warning("No sampler found. Assuming reparametrization trick.")

        params = inspect.signature(self.decoder.forward).parameters
        dec_in = list(zip(*[mirtp.force_list(io.decoder[prop]) for prop in params]))
        dec_out = mirtp.split_as(mirtp.resolve_io_config(io.decoder["out"]), dec_in)

        self.dec_res_fill = [
            mirtp.get_result_fillers(self.decoder, out) for out in dec_out
        ]
        get_dec_filler = iter(self.dec_res_fill)

        for keys in dec_in:
            self.dec_fwds.append(
                lambda td, tk=keys, args=params.keys(), dec=self.decoder, filler=next(
                    get_dec_filler
                ): filler(
                    td,
                    dec(
                        **dict(
                            zip(
                                args,
                                list(
                                    td[k] if type(k) is str else list(td[j] for j in k)
                                    for k in tk
                                ),
                            )
                        )
                    ),
                )
            )

        params = inspect.signature(self.decoder.forward).parameters
        if io.generator is not None:
            gen_in = list(
                zip(*[mirtp.force_list(io.generator[prop]) for prop in params])
            )
            gen_out = mirtp.split_as(
                mirtp.resolve_io_config(io.generator["out"]), gen_in
            )

            self.gen_res_fill = [
                mirtp.get_result_fillers(self.decoder, out) for out in gen_out
            ]
            get_gen_filler = iter(self.gen_res_fill)

            for keys in gen_in:
                self.gen_fwds.append(
                    lambda td, tk=keys, args=params.keys(), dec=self.decoder, filler=next(
                        get_gen_filler
                    ): filler(
                        td,
                        dec(
                            **dict(
                                zip(
                                    args,
                                    list(
                                        (
                                            td[k]
                                            if type(k) is str
                                            else list(td[j] for j in k)
                                        )
                                        for k in tk
                                    ),
                                )
                            )
                        ),
                    )
                )
        else:
            self.gen_fwds.append(torch.nn.Identity())

        # changes to work with PTL 2.0
        self.validation_step_outputs = []

    def forward(
        self, td: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        for e, f, r, d in zip(
            self.enc_fwds, self.f2mu_std_fwds, self.rep_fwds, self.dec_fwds
        ):
            e(td)
            f(td)
            r(td)
            d(td)
        return td

    def validation_step(
        self,
        batch: typing.Dict[str, torch.Tensor],
        batch_nb: int,
        dataloader_index: int = 0,
    ) -> dict:
        preprocessed = self.preprocess(batch)
        prediction = self(preprocessed)
        outputs = self.postprocess(prediction)
        outputs = self.sample(outputs)  # generation block starts
        [d(outputs) for d in self.gen_fwds]
        generated = self.generate(outputs)
        metrics = self.validation(outputs)
        gen_metrics = self.gen_validation(generated)
        aggregated = {
            "metrics": toolz.merge(
                metrics, {"gen_" + str(k): v for k, v in gen_metrics.items()}
            )
        }  # generation block ends
        aggregated = toolz.merge(
            aggregated,
            {
                "td": toolz.merge(
                    generated, {"__moai__": {"epoch": self.trainer.current_epoch}}
                )
            },
        )
        self.validation_step_outputs.append(aggregated)
        return aggregated

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        # latent space walking
        sampled = self.walk(outputs[0]["td"])
        [d(sampled) for d in self.gen_fwds]
        generated = self.generate(sampled)
        generated["__moai__"]["isVal"] = True
        [vis(generated) for vis in self.visualization.latent_visualizers]

        list_of_outputs = (
            [outputs] if isinstance(toolz.get([0, 0], outputs)[0], dict) else outputs
        )
        all_metrics = defaultdict(list)
        for i, o in enumerate(list_of_outputs):
            keys = next(iter(o), {})["metrics"].keys()
            metrics = {}
            for key in keys:
                if key[:4] == "gen_":
                    metrics[key] = np.mean(
                        np.array(  # TODO remove mean adapt logging
                            [d["metrics"][key].item() for d in o if key in d["metrics"]]
                        )
                    )
                else:
                    metrics[key] = np.mean(
                        np.array(
                            [d["metrics"][key].item() for d in o if key in d["metrics"]]
                        )
                    )
                all_metrics[key].append(metrics[key])
            metrics = toolz.keymap(
                lambda k: k.strip("gen_"), metrics
            )  # remove hardcoded 'gen_' prefix for logging
            all_metrics = toolz.keymap(
                lambda k: k.strip("gen_"), all_metrics
            )  # remove hardcoded 'gen_' prefix for logging
            log_metrics = toolz.keymap(
                lambda k: f"val_{k}/{list(self.data.val.iterator.datasets.keys())[i]}",
                metrics,
            )
            self.log_dict(
                log_metrics, prog_bar=False, logger=True, on_epoch=True, sync_dist=True
            )
        all_metrics = toolz.valmap(lambda v: sum(v) / len(v), all_metrics)
        self.validation_step_outputs.clear()  # free memory
        self.log_dict(
            all_metrics, prog_bar=True, logger=False, on_epoch=True, sync_dist=True
        )


class Feature2MuStd(torch.nn.Module):
    def __init__(
        self,
        configuration: omegaconf.DictConfig,
    ) -> None:
        super(Feature2MuStd, self).__init__()

        self.flatten = (
            hyu.instantiate(configuration.flatten)
            if configuration.flatten is not None
            else torch.nn.Flatten()
        )

        self.linear_mu = mil.make_linear_block(
            block_type=configuration.linear.type,
            linear_type="linear",
            activation_type=configuration.linear.activation.type,
            in_features=configuration.features_head.enc_out_dim,
            out_features=configuration.features_head.latent_dim,
        )
        self.linear_logvar = mil.make_linear_block(
            block_type=configuration.linear.type,
            linear_type="linear",
            activation_type=configuration.linear.activation.type,
            in_features=configuration.features_head.enc_out_dim,
            out_features=configuration.features_head.latent_dim,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x_flat = self.flatten(features)
        mu = self.linear_mu(x_flat)
        logvar = self.linear_logvar(x_flat)

        return mu, logvar


class Reparametrizer(torch.nn.Module):
    def __init__(
        self,
        prior: torch.nn.Module,
    ) -> None:
        super(Reparametrizer, self).__init__()
        self.reparametrize = prior

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        # x_enc: torch.Tensor,
    ) -> torch.Tensor:  # typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.reparametrize(mu, logvar)
        # v, sigma, z = self.reparametrize(x_enc)
        # return v, sigma, z


class ControlledReparametrizer(Reparametrizer):
    def __init__(self, prior: torch.nn.Module) -> None:
        super().__init__(prior)

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        ctrl: torch.Tensor,
    ) -> torch.Tensor:  # typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return torch.cat((self.reparametrize(mu, logvar), ctrl), dim=-1)
