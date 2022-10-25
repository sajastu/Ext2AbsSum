import math
import os
import sys
import time
import warnings
from typing import Dict, Union, Any, List, Optional, Tuple, NamedTuple

# import numpy as np
import numpy as np
import torch
from torch import nn
from torch.cuda import amp

from torch.utils.data import DataLoader, DistributedSampler

from GreaseLM.grease_model.utils import optimization_utils
from transformers import Seq2SeqTrainer, get_scheduler, is_torch_tpu_available, WEIGHTS_NAME, PretrainedConfig, \
    CONFIG_NAME, __version__, TrainerState
from transformers.debug_utils import DebugOption
from transformers.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init, deepspeed_reinit
import torch.distributed as dist

from transformers.utils.logging import tqdm

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from transformers.integrations import hp_params
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_pt_utils import IterableDatasetShard, find_batch_size, nested_concat, nested_numpify, \
    nested_truncate
from transformers.trainer_utils import TrainOutput, speed_metrics, set_seed, get_last_checkpoint, has_length, \
    ShardedDDPOption, HPSearchBackend, EvalLoopOutput, denumpify_detensorize
from transformers.utils import logging

logger = logging.get_logger(__name__)

class TGSumTrainer(Seq2SeqTrainer):

    def __init__(
            self,
            model=None,
            loading_info=None,
            args=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            data_collator=None,
            compute_metrics=None
    ):

        self.loading_info = loading_info
        super(TGSumTrainer, self).__init__(
            model=model,
            args=args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if args.predict_with_generate else None,
        )

    def sep_params(self, model, not_loaded_param_names):
        small_lr_params = dict()
        large_lr_params = dict()
        for n, p in model.named_parameters():
            if n not in ['model.' + x for x in not_loaded_param_names]:
                small_lr_params[n] = p
            else:
                large_lr_params[n] = p
        return small_lr_params, large_lr_params






    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # loaded_model_keys = [p for p in self.model.state_dict().keys() if p.replace('model.','') not in self.loading_info['missing_keys']]
        # loaded_params, not_loaded_params, params_to_freeze, small_lr_params, large_lr_params = self.sep_params(loaded_model_keys)
        # loaded_params, not_loaded_params, small_lr_params, large_lr_params = self.sep_params2()
        # small_lr_params, large_lr_params = self.sep_params(self.model, self.loading_info['missing_keys'])
        small_lr_params = {}
        large_lr_params = {}
        for n, p in self.model.named_parameters():
            if n in self.loading_info['graph_keys']:
                large_lr_params[n] = p
            else:
                small_lr_params[n] = p

        print('Non-loaded parameters:')
        for name, param in large_lr_params.items():
            if param.requires_grad:
                print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
            else:
                print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))

        # if self.optimizer is None:
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in small_lr_params.items() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate},
            {'params': [p for n, p in small_lr_params.items() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.learning_rate},
            {'params': [p for n, p in large_lr_params.items() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': 5e-5},
            {'params': [p for n, p in large_lr_params.items() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': 5e-5},

        ]

        optimizer_cls, optimizer_kwargs = TGSumTrainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        # self.graph_optimizer = optimization_utils.OPTIMIZER_CLASSES['radam'](optimizer_grouped_parameters_graph)

        return self.optimizer

    # def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    #     """
    #     Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    #     passed as an argument.
    #
    #     Args:
    #         num_training_steps (int): The number of training steps to do.
    #     """
    #     self.lr_scheduler = get_scheduler(
    #         self.args.lr_scheduler_type,
    #         optimizer=self.optimizer if optimizer is None else optimizer,
    #         num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
    #         num_training_steps=num_training_steps,
    #     )
    #     # try:
    #     #     from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
    #     # except:
    #     #     from transformers import get_constant_schedule, get_constant_schedule_with_warmup, \
    #     #         get_linear_schedule_with_warmup
    #
    #     self.graph_scheduler = get_scheduler(
    #         'polynomial',
    #         optimizer=self.graph_optimizer,
    #         num_warmup_steps=4000,
    #         num_training_steps=num_training_steps,
    #     )
    #
    #         # self.graph_scheduler = get_constant_schedule(self.graph_optimizer)
    #
    #     return self.lr_scheduler

    # def train(
    #     self,
    #     resume_from_checkpoint: Optional[Union[str, bool]] = None,
    #     trial: Union["optuna.Trial", Dict[str, Any]] = None,
    #     ignore_keys_for_eval: Optional[List[str]] = None,
    #     **kwargs,
    # ):
    #     """
    #     Main training entry point.
    #
    #     Args:
    #         resume_from_checkpoint (`str` or `bool`, *optional*):
    #             If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
    #             `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
    #             of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
    #         trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
    #             The trial run or the hyperparameter dictionary for hyperparameter search.
    #         ignore_keys_for_eval (`List[str]`, *optional*)
    #             A list of keys in the output of your model (if it is a dictionary) that should be ignored when
    #             gathering predictions for evaluation during the training.
    #         kwargs:
    #             Additional keyword arguments used to hide deprecated arguments
    #     """
    #     resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint
    #
    #     # memory metrics - must set up as early as possible
    #     self._memory_tracker.start()
    #
    #     args = self.args
    #
    #     self.is_in_train = True
    #
    #     # do_train is not a reliable argument, as it might not be set and .train() still called, so
    #     # the following is a workaround:
    #     if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
    #         self._move_model_to_device(self.model, args.device)
    #
    #     if "model_path" in kwargs:
    #         resume_from_checkpoint = kwargs.pop("model_path")
    #         warnings.warn(
    #             "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
    #             "instead.",
    #             FutureWarning,
    #         )
    #     if len(kwargs) > 0:
    #         raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
    #     # This might change the seed so needs to run first.
    #     self._hp_search_setup(trial)
    #
    #     # Model re-init
    #     model_reloaded = False
    #     if self.model_init is not None:
    #         # Seed must be set before instantiating the model when using model_init.
    #         set_seed(args.seed)
    #         self.model = self.call_model_init(trial)
    #         model_reloaded = True
    #         # Reinitializes optimizer and scheduler
    #         # self.optimizer, self.lr_scheduler, self.graph_optimizer, self.graph_scheduler = None, None, None, None
    #         self.optimizer, self.lr_scheduler = None, None
    #
    #     # Load potential model checkpoint
    #     if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
    #         resume_from_checkpoint = get_last_checkpoint(args.output_dir)
    #         if resume_from_checkpoint is None:
    #             raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
    #
    #     if resume_from_checkpoint is not None:
    #         if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
    #             raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")
    #
    #         logger.info(f"Loading model from {resume_from_checkpoint}).")
    #
    #         if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
    #             config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
    #             checkpoint_version = config.transformers_version
    #             if checkpoint_version is not None and checkpoint_version != __version__:
    #                 logger.warning(
    #                     f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
    #                     f"Transformers but your current version is {__version__}. This is not recommended and could "
    #                     "yield to errors or unwanted behaviors."
    #                 )
    #
    #         if args.deepspeed:
    #             # will be resumed in deepspeed_init
    #             pass
    #         else:
    #             # We load the model state dict on the CPU to avoid an OOM error.
    #             state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
    #             # If the model is on the GPU, it still works!
    #             self._load_state_dict_in_model(state_dict)
    #
    #             # release memory
    #             del state_dict
    #
    #     # If model was re-initialized, put it on the right device and update self.model_wrapped
    #     if model_reloaded:
    #         if self.place_model_on_device:
    #             self._move_model_to_device(self.model, args.device)
    #         self.model_wrapped = self.model
    #
    #     # Keeping track whether we can can len() on the dataset or not
    #     train_dataset_is_sized = has_length(self.train_dataset)
    #
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()
    #
    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
    #     if train_dataset_is_sized:
    #         num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #         else:
    #             max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = len(self.train_dataset) * args.num_train_epochs
    #     else:
    #         # see __init__. max_steps is set when the dataset has no __len__
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #
    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa
    #
    #     delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
    #     if args.deepspeed:
    #         deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
    #             self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine
    #         self.optimizer = optimizer
    #         self.lr_scheduler = lr_scheduler
    #     elif not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    #
    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None
    #
    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable()
    #
    #     model = self._wrap_model(self.model_wrapped)
    #
    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model
    #
    #     if delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    #
    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)
    #
    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.
    #
    #     # Train!
    #     num_examples = (
    #         self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
    #     )
    #
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples}")
    #     logger.info(f"  Num Epochs = {num_train_epochs}")
    #     logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    #     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps}")
    #
    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None
    #
    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0
    #
    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             logger.info(
    #                 f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
    #                 "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
    #                 "flag to your launch command, but you will resume the training on data already seen by your model."
    #             )
    #             if self.is_local_process_zero() and not args.disable_tqdm:
    #                 steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
    #                 steps_trained_progress_bar.set_description("Skipping the first batches")
    #
    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     # self.callback_handler.graph_optimizer = self.graph_optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     # self.callback_handler.graph_scheduler = self.graph_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
    #     if trial is not None:
    #         assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()
    #
    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()
    #
    #     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
    #
    #     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    #     if not args.ignore_data_skip:
    #         for epoch in range(epochs_trained):
    #             # We just need to begin an iteration to create the randomization of the sampler.
    #             for _ in train_dataloader:
    #                 break
    #
    #     for epoch in range(epochs_trained, num_train_epochs):
    #         if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
    #             train_dataloader.sampler.set_epoch(epoch)
    #         elif isinstance(train_dataloader.dataset, IterableDatasetShard):
    #             train_dataloader.dataset.set_epoch(epoch)
    #
    #         if is_torch_tpu_available():
    #             parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
    #             epoch_iterator = parallel_loader
    #         else:
    #             epoch_iterator = train_dataloader
    #
    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None
    #
    #         steps_in_epoch = (
    #             len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
    #
    #         step = -1
    #         for step, inputs in enumerate(epoch_iterator):
    #
    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 if steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.update(1)
    #                 if steps_trained_in_current_epoch == 0:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                 continue
    #             elif steps_trained_progress_bar is not None:
    #                 steps_trained_progress_bar.close()
    #                 steps_trained_progress_bar = None
    #
    #             if step % args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
    #
    #             if (
    #                 ((step + 1) % args.gradient_accumulation_steps != 0)
    #                 and args.local_rank != -1
    #                 and args._no_sync_in_gradient_accumulation
    #             ):
    #                 # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
    #                 with model.no_sync():
    #                     tr_loss_step = self.training_step(model, inputs)
    #             else:
    #                 tr_loss_step = self.training_step(model, inputs)
    #
    #             if (
    #                 args.logging_nan_inf_filter
    #                 and not is_torch_tpu_available()
    #                 and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #             ):
    #                 # if loss is nan or inf simply add the average of previous logged losses
    #                 tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
    #             else:
    #                 tr_loss += tr_loss_step
    #
    #             self.current_flos += float(self.floating_point_ops(inputs))
    #
    #             # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
    #             if self.deepspeed:
    #                 self.deepspeed.step()
    #
    #             if (step + 1) % args.gradient_accumulation_steps == 0 or (
    #                 # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                 steps_in_epoch <= args.gradient_accumulation_steps
    #                 and (step + 1) == steps_in_epoch
    #             ):
    #                 # Gradient clipping
    #                 if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
    #                     # deepspeed does its own clipping
    #
    #                     if self.do_grad_scaling:
    #                         # Reduce gradients first for XLA
    #                         if is_torch_tpu_available():
    #                             gradients = xm._fetch_gradients(self.optimizer)
    #                             # gradients_graph = xm._fetch_gradients(self.graph_optimizer)
    #                             xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
    #                             # xm.all_reduce("sum", gradients_graph, scale=1.0 / xm.xrt_world_size())
    #                         # AMP: gradients need unscaling
    #                         self.scaler.unscale_(self.optimizer)
    #                         # self.scaler.unscale_(self.graph_optimizer)
    #
    #                     if hasattr(self.optimizer, "clip_grad_norm"):
    #                         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
    #                         self.optimizer.clip_grad_norm(args.max_grad_norm)
    #                         # self.graph_optimizer.clip_grad_norm(args.max_grad_norm)
    #                     elif hasattr(model, "clip_grad_norm_"):
    #                         # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
    #                         model.clip_grad_norm_(args.max_grad_norm)
    #                     else:
    #                         # Revert to normal clipping otherwise, handling Apex or full precision
    #                         nn.utils.clip_grad_norm_(
    #                             amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
    #                             args.max_grad_norm,
    #                         )
    #                         # nn.utils.clip_grad_norm_(
    #                         #     amp.master_params(self.graph_optimizer) if self.use_apex else model.parameters(),
    #                         #     args.max_grad_norm,
    #                         # )
    #
    #                 # Optimizer step
    #                 optimizer_was_run = True
    #                 if self.deepspeed:
    #                     pass  # called outside the loop
    #                 elif is_torch_tpu_available():
    #                     if self.do_grad_scaling:
    #                         self.scaler.step(self.optimizer)
    #                         # self.scaler.step(self.graph_optimizer)
    #                         self.scaler.update()
    #                     else:
    #                         xm.optimizer_step(self.optimizer)
    #                         # xm.optimizer_step(self.graph_optimizer)
    #                 elif self.do_grad_scaling:
    #                     scale_before = self.scaler.get_scale()
    #                     self.scaler.step(self.optimizer)
    #                     # self.scaler.step(self.graph_optimizer)
    #                     self.scaler.update()
    #                     scale_after = self.scaler.get_scale()
    #                     optimizer_was_run = scale_before <= scale_after
    #                 else:
    #                     self.optimizer.step()
    #                     # self.graph_optimizer.step()
    #
    #                 if optimizer_was_run and not self.deepspeed:
    #                     self.lr_scheduler.step()
    #                     # self.graph_scheduler.step()
    #
    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = epoch + (step + 1) / steps_in_epoch
    #                 self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    #
    #                 self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
    #             else:
    #                 self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
    #
    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True
    #
    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
    #
    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_tpu_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break
    #
    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")
    #
    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sur the model has been saved by process 0.
    #         if is_torch_tpu_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         elif args.local_rank != -1:
    #             dist.barrier()
    #
    #         logger.info(
    #             f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
    #         )
    #
    #         best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
    #         if os.path.exists(best_model_path):
    #             if self.deepspeed:
    #                 # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
    #                 deepspeed_engine, optimizer, lr_scheduler = deepspeed_reinit(self)
    #                 self.model = deepspeed_engine.module
    #                 self.model_wrapped = deepspeed_engine
    #                 self.deepspeed = deepspeed_engine
    #                 self.optimizer = optimizer
    #                 self.lr_scheduler = lr_scheduler
    #                 self.deepspeed.load_checkpoint(
    #                     self.state.best_model_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
    #                 )
    #             else:
    #                 # We load the model state dict on the CPU to avoid an OOM error.
    #                 state_dict = torch.load(best_model_path, map_location="cpu")
    #                 # If the model is on the GPU, it still works!
    #                 self._load_state_dict_in_model(state_dict)
    #         else:
    #             logger.warning(
    #                 f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
    #                 "on multiple nodes, you should activate `--save_on_each_node`."
    #             )
    #
    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     train_loss = self._total_loss_scalar / self.state.global_step
    #
    #     metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss
    #
    #     self.is_in_train = False
    #
    #     self._memory_tracker.stop_and_update_metrics(metrics)
    #
    #     self.log(metrics)
    #
    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)
    #
    #     return TrainOutput(self.state.global_step, train_loss, metrics)

    # def log(self, logs: Dict[str, float]) -> None:
    #     """
    #     Log `logs` on the various objects watching training.
    #
    #     Subclass and override this method to inject custom behavior.
    #
    #     Args:
    #         logs (`Dict[str, float]`):
    #             The values to log.
    #     """
    #     if self.state.epoch is not None:
    #         logs["epoch"] = round(self.state.epoch, 2)
    #
    #     # add graph-specific logs...
    #     logs["graph_lr"] = self.graph_scheduler.get_last_lr()[0]
    #
    #     output = {**logs, **{"step": self.state.global_step}}
    #     self.state.log_history.append(output)
    #     self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            subgraphs=inputs['subgraphs'],
            doc_ids=inputs['doc_ids'],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        import pdb;pdb.set_trace()
        return (loss, generated_tokens, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset
        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        ids_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            import pdb;pdb.set_trace()
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host

            if 'doc_ids' in inputs.keys():
                ids_host = inputs['doc_ids'] if ids_host is None else ids_host + inputs['doc_ids']
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:

                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU

        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels, doc_ids=ids_host))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]
    doc_ids: Optional[Union[str]]