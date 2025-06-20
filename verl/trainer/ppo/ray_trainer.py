# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, List

import re
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayResourcePool,
    RayWorkerGroup,
    RayClassWithInitArgs,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.response_checker import error_response, contains_answer

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"
):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = (
        data.batch["info_mask"]
        if "info_mask" in data.batch
        else data.batch["attention_mask"]
    )
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_prob"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == "gae":
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "grpo":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5))
                .detach()
                .item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(
            torch.eq(response_length, max_response_length).float()
        )
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(
            torch.eq(prompt_length, max_prompt_length).float()
        )
        .detach()
        .item(),
    }

    # metrics for actions
    if "turns_stats" in batch.meta_info:
        metrics["env/number_of_actions/mean"] = float(
            np.array(batch.meta_info["turns_stats"], dtype=np.int16).mean()
        )
        metrics["env/number_of_actions/max"] = float(
            np.array(batch.meta_info["turns_stats"], dtype=np.int16).max()
        )
        metrics["env/number_of_actions/min"] = float(
            np.array(batch.meta_info["turns_stats"], dtype=np.int16).min()
        )
    if "active_mask" in batch.meta_info:
        metrics["env/finish_ratio"] = 1 - float(
            np.array(batch.meta_info["active_mask"], dtype=np.int16).mean()
        )
    if "valid_action_stats" in batch.meta_info:
        metrics["env/number_of_valid_action"] = float(
            np.array(batch.meta_info["valid_action_stats"], dtype=np.int16).mean()
        )
        metrics["env/ratio_of_valid_action"] = float(
            (
                np.array(batch.meta_info["valid_action_stats"], dtype=np.int16)
                / np.array(batch.meta_info["turns_stats"], dtype=np.int16)
            ).mean()
        )
    if "valid_search_stats" in batch.meta_info:
        metrics["env/number_of_valid_search"] = float(
            np.array(batch.meta_info["valid_search_stats"], dtype=np.int16).mean()
        )

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{
            name: num_overall_tokens
            for name in [
                "ref",
                "values",
                "adv",
                "update_critic",
                "update_actor",
                "rollout",
            ]
        },
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name]
            * 1000
            / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == "fixed":
                self.kl_ctrl = core_algos.FixedKLController(
                    kl_coef=config.algorithm.kl_ctrl.kl_coef
                )
            elif config.algorithm.kl_ctrl.type == "adaptive":
                assert (
                    config.algorithm.kl_ctrl.horizon > 0
                ), f"horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}"
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        self._create_dataloader()
        self._init_logger()

        # Initialize training state
        self.current_epoch = 0
        self.current_step_in_epoch = 0

        # Initialize manual batch skipping attributes for fallback mechanism
        self._manual_skip_batches = 0
        self._manual_skip_enabled = False

    def _init_logger(self):
        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler

        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(
                    f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}"
                )
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(
                    self.config.data.train_data_num, random_state=42
                )
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        # Create sampler for shuffling if needed
        if self.config.data.shuffle_train_dataloader:
            sampler = RandomSampler(self.train_dataset)
        else:
            sampler = None

        # Use StatefulDataLoader for training data to support checkpoint resume
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(
                    f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}"
                )
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(
                    self.config.data.val_data_num, random_state=42
                )
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        # Use regular DataLoader for validation (no need for stateful behavior)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch

        reward_tensor_lst = []
        data_source_lst = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_engine=self.config.retriever.search_engine,
            search_url=self.config.retriever.url,
            topk=self.config.retriever.topk,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if (
                    self.config.reward_model.enable
                    and test_batch[0].non_tensor_batch["reward_model"]["style"]
                    == "model"
                ):
                    return {}

                test_gen_batch = test_batch.pop(
                    ["input_ids", "attention_mask", "position_ids"]
                )
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": False,
                    "validate": True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                    test_gen_batch, self.actor_rollout_wg.world_size
                )
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                    test_gen_batch_padded
                )
                # unpad
                test_output_gen_batch = unpad_dataproto(
                    test_output_gen_batch_padded, pad_size=pad_size
                )
                print("validation generation end")

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(
                    test_batch.non_tensor_batch.get(
                        "data_source", ["unknown"] * reward_tensor.shape[0]
                    )
                )
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"]
                )
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": False,
                    "validate": True,
                }
                with _timer("step", timing_raw):
                    first_input_ids = test_gen_batch.batch["input_ids"][
                        :, -gen_config.max_start_length :
                    ].clone()
                    with _timer("gen", timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )

                    test_batch = test_batch.union(final_gen_batch_output)

                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()

                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(
                        test_batch.non_tensor_batch.get(
                            "data_source", ["unknown"] * reward_tensor.shape[0]
                        )
                    )

        reward_tensor = torch.cat(
            [rw.sum(-1) for rw in reward_tensor_lst], dim=0
        ).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == "gae":
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
            self.use_critic = True

        elif self.config.algorithm.adv_estimator == "grpo":
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # Create checkpoint directory with global step
        checkpoint_dir = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save actor model
        actor_local_path = os.path.join(checkpoint_dir, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        # Save critic model if available
        if self.use_critic:
            critic_local_path = os.path.join(checkpoint_dir, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

        # Save training state
        self._save_training_state(checkpoint_dir)

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        print(
            f"Checkpoint saved at global step {self.global_steps} to {checkpoint_dir}"
        )

        # Clean up old checkpoints if enabled
        if self.config.trainer.get("checkpoint_cleanup", False):
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints, keeping only the most recent ones"""
        max_checkpoints = self.config.trainer.get("max_checkpoints_to_keep", 5)

        if not os.path.exists(self.config.trainer.default_local_dir):
            return

        # Find all checkpoint directories
        checkpoint_dirs = []
        for item in os.listdir(self.config.trainer.default_local_dir):
            if item.startswith("global_step_"):
                try:
                    step = int(item.split("_")[-1])
                    checkpoint_path = os.path.join(
                        self.config.trainer.default_local_dir, item
                    )
                    checkpoint_dirs.append((step, checkpoint_path))
                except ValueError:
                    continue

        # Sort by step number and keep only the most recent ones
        if len(checkpoint_dirs) > max_checkpoints:
            checkpoint_dirs.sort(key=lambda x: x[0])
            checkpoints_to_remove = checkpoint_dirs[:-max_checkpoints]

            for step, checkpoint_path in checkpoints_to_remove:
                try:
                    import shutil

                    shutil.rmtree(checkpoint_path)
                    print(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    print(f"Failed to remove checkpoint {checkpoint_path}: {e}")

        print(
            f"Checkpoint cleanup completed. Keeping {min(len(checkpoint_dirs), max_checkpoints)} checkpoints."
        )

    def _save_training_state(self, checkpoint_dir: str):
        """Save training state including global steps, epoch, and KL controller state"""
        import json
        import time

        training_state = {
            "global_steps": self.global_steps,
            "current_epoch": getattr(self, "current_epoch", 0),
            "current_step_in_epoch": getattr(self, "current_step_in_epoch", 0),
            "kl_ctrl_state": (
                self.kl_ctrl.get_state() if hasattr(self.kl_ctrl, "get_state") else None
            ),
            "total_training_steps": self.total_training_steps,
            "dataloader_metadata": {
                "total_batches_per_epoch": len(self.train_dataloader),
                "batch_size": self.config.data.train_batch_size,
                "dataset_size": len(self.train_dataloader.dataset),
                "samples_processed_in_current_epoch": getattr(
                    self, "current_step_in_epoch", 0
                )
                * self.config.data.train_batch_size,
            },
        }

        state_path = os.path.join(checkpoint_dir, "training_state.json")
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)

        dataloader_local_path = os.path.join(checkpoint_dir, "dataloader_state.pt")
        dataloader_success = False
        try:
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"DataLoader state saved to {dataloader_local_path}")
            dataloader_success = True
        except Exception as e:
            print(f"Warning: Failed to save DataLoader state: {e}")
            failed_marker_path = os.path.join(
                checkpoint_dir, "dataloader_save_failed.marker"
            )
            with open(failed_marker_path, "w") as f:
                f.write(f"DataLoader state save failed: {str(e)}\n")
                f.write(f"Timestamp: {time.time()}\n")

        training_state["dataloader_state_saved"] = dataloader_success
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)

        print(f"Training state saved to {state_path}")

    def _load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint from specified directory"""
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory {checkpoint_dir} does not exist")
            return False

        # Load training state
        if not self._load_training_state(checkpoint_dir):
            return False

        # Load actor model
        actor_path = os.path.join(checkpoint_dir, "actor")
        if os.path.exists(actor_path):
            self.actor_rollout_wg.load_checkpoint(actor_path)
            print(f"Actor model loaded from {actor_path}")
        else:
            print(f"Actor checkpoint not found at {actor_path}")
            return False

        # Load critic model if available
        if self.use_critic:
            critic_path = os.path.join(checkpoint_dir, "critic")
            if os.path.exists(critic_path):
                self.critic_wg.load_checkpoint(critic_path)
                print(f"Critic model loaded from {critic_path}")
            else:
                print(f"Critic checkpoint not found at {critic_path}")
                return False

        print(f"Checkpoint loaded from {checkpoint_dir}")
        return True

    def _load_training_state(self, checkpoint_dir: str):
        """Load training state with enhanced DataLoader fallback mechanism"""
        import json
        import time

        state_path = os.path.join(checkpoint_dir, "training_state.json")
        if not os.path.exists(state_path):
            print(f"Training state file not found at {state_path}")
            return False

        with open(state_path, "r") as f:
            training_state = json.load(f)

        self.global_steps = training_state["global_steps"]
        self.current_epoch = training_state.get("current_epoch", 0)
        self.current_step_in_epoch = training_state.get("current_step_in_epoch", 0)

        # Restore KL controller state if available
        if training_state.get("kl_ctrl_state") and hasattr(self.kl_ctrl, "set_state"):
            self.kl_ctrl.set_state(training_state["kl_ctrl_state"])

        # Enhanced DataLoader state loading with fallback mechanism
        dataloader_loaded = self._load_dataloader_state_with_fallback(
            checkpoint_dir, training_state
        )

        if not dataloader_loaded:
            print(
                "Warning: DataLoader state could not be restored. Using fallback mechanism."
            )
            # Set up manual batch skipping
            self._setup_manual_batch_skipping(training_state)

        print(
            f"Training state loaded: global_steps={self.global_steps}, epoch={self.current_epoch}, step_in_epoch={self.current_step_in_epoch}"
        )
        return True

    def _load_dataloader_state_with_fallback(
        self, checkpoint_dir: str, training_state: dict
    ) -> bool:
        """Load DataLoader state with multiple fallback strategies"""
        dataloader_state_path = os.path.join(checkpoint_dir, "dataloader_state.pt")

        # Check if DataLoader state was successfully saved
        dataloader_state_saved = training_state.get(
            "dataloader_state_saved", True
        )  # Default to True for backward compatibility

        if not dataloader_state_saved:
            print(
                "DataLoader state was not saved successfully during checkpoint creation"
            )
            return False

        if not os.path.exists(dataloader_state_path):
            print(f"DataLoader state file not found at {dataloader_state_path}")
            return False

        # Try to load DataLoader state with multiple attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                dataloader_state_dict = torch.load(
                    dataloader_state_path, map_location="cpu"
                )

                # Validate the state dict before loading
                if self._validate_dataloader_state(
                    dataloader_state_dict, training_state
                ):
                    self.train_dataloader.load_state_dict(dataloader_state_dict)
                    print(
                        f"DataLoader state loaded successfully from {dataloader_state_path}"
                    )
                    return True
                else:
                    print(
                        f"DataLoader state validation failed on attempt {attempt + 1}"
                    )

            except Exception as e:
                print(f"Attempt {attempt + 1} failed to load DataLoader state: {e}")
                if attempt < max_attempts - 1:
                    print(f"Retrying... ({attempt + 2}/{max_attempts})")
                    time.sleep(1)  # Brief pause before retry

        return False

    def _validate_dataloader_state(
        self, state_dict: dict, training_state: dict
    ) -> bool:
        """Validate DataLoader state dict for consistency"""
        try:
            # Basic validation - check if state dict has expected structure
            if not isinstance(state_dict, dict):
                print("DataLoader state is not a dictionary")
                return False

            # Check if metadata matches (if available)
            metadata = training_state.get("dataloader_metadata", {})
            if metadata:
                expected_batch_size = metadata.get("batch_size")
                expected_dataset_size = metadata.get("dataset_size")

                # Add more specific validations based on StatefulDataLoader structure
                # This is a placeholder - actual validation depends on StatefulDataLoader implementation

            return True

        except Exception as e:
            print(f"DataLoader state validation error: {e}")
            return False

    def _setup_manual_batch_skipping(self, training_state: dict):
        """Setup manual batch skipping when DataLoader state cannot be restored"""
        metadata = training_state.get("dataloader_metadata", {})

        if metadata:
            samples_to_skip = metadata.get("samples_processed_in_current_epoch", 0)
            batches_to_skip = samples_to_skip // self.config.data.train_batch_size

            # Store skipping information for use in training loop
            self._manual_skip_batches = batches_to_skip
            self._manual_skip_enabled = True

            print(
                f"Manual batch skipping enabled: will skip {batches_to_skip} batches in current epoch"
            )
            print(f"This corresponds to {samples_to_skip} samples already processed")
        else:
            # Fallback to epoch-level skipping if no metadata available
            self._manual_skip_batches = 0
            self._manual_skip_enabled = False
            print(
                "No dataloader metadata available. Training will restart from beginning of current epoch"
            )

    def _should_skip_batch(self, batch_idx: int) -> bool:
        """Check if current batch should be skipped due to manual skipping"""
        if not getattr(self, "_manual_skip_enabled", False):
            return False

        skip_count = getattr(self, "_manual_skip_batches", 0)
        if batch_idx < skip_count:
            return True

        # Disable manual skipping after we've skipped the required batches
        if batch_idx == skip_count:
            self._manual_skip_enabled = False
            print(f"Finished skipping {skip_count} batches. Resuming normal training.")

        return False

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint directory"""
        if not os.path.exists(self.config.trainer.default_local_dir):
            return None

        checkpoint_dirs = []
        for item in os.listdir(self.config.trainer.default_local_dir):
            if item.startswith("global_step_"):
                try:
                    step = int(item.split("_")[-1])
                    checkpoint_dirs.append(
                        (
                            step,
                            os.path.join(self.config.trainer.default_local_dir, item),
                        )
                    )
                except ValueError:
                    continue

        if not checkpoint_dirs:
            return None

        # Return the directory with the highest global step
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs[-1][1]

    def resume_from_checkpoint(self, checkpoint_dir: str = None):
        """Resume training from checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = self._find_latest_checkpoint()

        if checkpoint_dir is None:
            print("No checkpoint found, starting training from scratch")
            return False

        print(f"Resuming from checkpoint: {checkpoint_dir}")
        return self._load_checkpoint(checkpoint_dir)

    def resume_from_checkpoint_with_fsdp_manager(self, checkpoint_root: str = None):
        """
        Resume training from checkpoint using FSDPCheckpointManager.
        This method provides more robust checkpoint handling with FSDP.

        Args:
            checkpoint_root: Root directory containing checkpoints. If None, uses default_local_dir.

        Returns:
            bool: True if successfully resumed, False otherwise
        """
        if checkpoint_root is None:
            checkpoint_root = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_root):
            print(f"Checkpoint root directory {checkpoint_root} does not exist")
            return False

        try:
            # Use FSDPCheckpointManager to resume actor
            from verl.utils.checkpoint.fsdp_checkpoint_manager import (
                FSDPCheckpointManager,
            )

            # Create a dummy checkpoint manager to find the latest checkpoint
            # We'll use the actual managers in the workers for loading
            latest_ckpt_path = self._find_latest_checkpoint_in_root(checkpoint_root)

            if latest_ckpt_path is None:
                print("No checkpoint found for resume, starting from scratch")
                return False

            # Extract global step from path
            try:
                global_step = int(os.path.basename(latest_ckpt_path).split("_")[-1])
            except ValueError:
                print(f"Could not parse global step from {latest_ckpt_path}")
                return False

            # Load training state first
            if not self._load_training_state(latest_ckpt_path):
                print("Failed to load training state")
                return False

            # Load actor model using worker's FSDPCheckpointManager
            actor_path = os.path.join(latest_ckpt_path, "actor")
            if os.path.exists(actor_path):
                self.actor_rollout_wg.load_checkpoint(actor_path)
                print(f"Actor model resumed from {actor_path}")
            else:
                print(f"Actor checkpoint not found at {actor_path}")
                return False

            # Load critic model if available
            if self.use_critic:
                critic_path = os.path.join(latest_ckpt_path, "critic")
                if os.path.exists(critic_path):
                    self.critic_wg.load_checkpoint(critic_path)
                    print(f"Critic model resumed from {critic_path}")
                else:
                    print(f"Critic checkpoint not found at {critic_path}")
                    return False

            # Update global step
            self.global_steps = global_step

            print(
                f"Successfully resumed training from checkpoint at global step {global_step}"
            )
            return True

        except Exception as e:
            print(f"Error during checkpoint resume: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _find_latest_checkpoint_in_root(self, checkpoint_root: str):
        """Find the latest checkpoint directory in the given root"""
        if not os.path.exists(checkpoint_root):
            return None

        checkpoint_dirs = []
        for item in os.listdir(checkpoint_root):
            if item.startswith("global_step_"):
                try:
                    step = int(item.split("_")[-1])
                    checkpoint_dirs.append(
                        (
                            step,
                            os.path.join(checkpoint_root, item),
                        )
                    )
                except ValueError:
                    continue

        if not checkpoint_dirs:
            return None

        # Return the directory with the highest global step
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs[-1][1]

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            attention_mask.view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def compute_response_quality_metrics(self, responses_str: List[str]) -> dict:
        """Compute response quality metrics."""
        contains_answer_list = [contains_answer(response) for response in responses_str]
        error_answer_list = [error_response(response) for response in responses_str]
        return {
            "response_quality/finish_ratio": float(sum(contains_answer_list)) / len(responses_str),
            "response_quality/error_ratio": float(sum(error_answer_list)) / len(responses_str),
        }

    def fit(self, resume_from_checkpoint: str = None):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger

        # Try to resume from checkpoint
        if resume_from_checkpoint is not None or self.config.trainer.get(
            "auto_resume", False
        ):
            if self.resume_from_checkpoint(resume_from_checkpoint):
                print(
                    f"Successfully resumed from checkpoint at global step {self.global_steps}"
                )
            else:
                print("Failed to resume from checkpoint, starting from scratch")
                self.global_steps = 0
        else:
            self.global_steps = 0

        # perform validation before training (skip if resuming and already past initial validation)
        # currently, we only support validation using the reward_function.
        if (
            self.val_reward_fn is not None
            and self.config.trainer.get("val_before_train", True)
            and self.global_steps == 0
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1 if not resuming
        if self.global_steps == 0:
            self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_engine=self.config.retriever.search_engine,
            search_url=self.config.retriever.url,
            topk=self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # start training loop
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.current_epoch = epoch
            # Only reset current_step_in_epoch if we're starting a new epoch (not resuming)
            if epoch > start_epoch:
                self.current_step_in_epoch = 0

            # StatefulDataLoader automatically handles resuming from the correct position
            # No need to create new dataloader or manually skip batches
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                # Check if we should skip this batch due to manual fallback mechanism
                if self._should_skip_batch(batch_idx):
                    print(
                        f"Skipping batch {batch_idx} (already processed before checkpoint)"
                    )
                    continue

                self.current_step_in_epoch = batch_idx
                print(f"epoch {epoch}, step {self.global_steps}, batch {batch_idx}")
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n_agent,
                    interleave=True,
                )

                # pop those keys for generation
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"]
                )

                ####################
                # original code here

                with _timer("step", timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch
                        )

                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                            dtype=object,
                        )
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n,
                            interleave=True,
                        )
                        batch = batch.union(gen_batch_output)

                    ####################
                    # Below is aLL about agents - the "LLM + forloop"
                    ####################
                    # with _timer('step', timing_raw):
                    else:
                        first_input_ids = (
                            gen_batch.batch["input_ids"][
                                :, -gen_config.max_start_length :
                            ]
                            .clone()
                            .long()
                        )

                        with _timer("gen", timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = (
                                final_gen_batch_output.batch[key].long()
                            )

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(
                                final_gen_batch_output
                            )
                            final_gen_batch_output = final_gen_batch_output.union(
                                output
                            )

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch["uid"] = batch.non_tensor_batch[
                            "index"
                        ].copy()

                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n,
                            interleave=True,
                        )
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != "old_log_probs":
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            if (
                                self.config.do_search
                                and self.config.actor_rollout_ref.actor.state_masking
                            ):
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if (
                        self.config.trainer.save_freq > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                responses_str = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                metrics.update(
                    self.compute_response_quality_metrics(responses_str)
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch["responses"].shape[-1]
        response_mask = batch.batch["attention_mask"][:, -response_length:]

        loss_mask = batch.batch["info_mask"][:, -response_length:]
        batch.batch["loss_mask"] = loss_mask

        metrics.update(
            {
                "state_tokens/total": loss_mask.sum().item(),
                "state_tokens/coverage": (loss_mask.sum() / response_mask.sum()).item(),
            }
        )

        return batch, metrics
