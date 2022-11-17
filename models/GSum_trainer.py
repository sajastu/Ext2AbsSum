import math
import os
import sys
import time
from distutils import dist
from typing import Dict, Union, Any, List, Optional, Tuple, NamedTuple

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from packaging import version

from transformers import Seq2SeqTrainer, is_torch_tpu_available, Trainer, TrainerState
from transformers.debug_utils import DebugOption
from transformers.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import is_fairscale_available, hp_params
from transformers.pipelines.base import Dataset
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, IterableDatasetShard, \
    nested_truncate
from transformers.trainer_utils import EvalLoopOutput, has_length, denumpify_detensorize, ShardedDDPOption, TrainOutput, \
    speed_metrics, HPSearchBackend, PredictionOutput
from transformers.utils import logging, is_sagemaker_mp_enabled

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

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


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        inputs['step'] = self.state.global_step

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
            # sect_loss = (outputs["sect_loss"] if isinstance(outputs, dict) else outputs[2])
            sent_loss = (outputs["sent_loss"] if isinstance(outputs, dict) else outputs[3])

        return (loss, outputs) if return_outputs else (loss, sent_loss)


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        sect_loss = torch.tensor(0.0).to(args.device)
        sent_loss = torch.tensor(0.0).to(args.device)
        topic_tr_loss = torch.tensor(0.0).to(args.device)
        lm_tr_loss = torch.tensor(0.0).to(args.device)

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            # steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)

            for step, inputs in enumerate(epoch_iterator):
                # import pdb;pdb.set_trace()
                # if step < 1190:
                #     continue
                # else:
                #     if steps_trained_progress_bar is not None:
                #         steps_trained_progress_bar.update(step)

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                # print(step)
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step, lm_loss_step, sent_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step, lm_loss_step, sent_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    sect_loss += sect_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    sent_loss += sent_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    topic_tr_loss += topic_tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    lm_tr_loss += lm_tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step
                    try:
                        sent_loss += sent_loss_step
                    except:
                        import pdb;pdb.set_trace()
                    # sect_loss += sect_loss_step
                    # topic_tr_loss += topic_loss_step
                    lm_tr_loss += lm_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, topic_tr_loss, lm_tr_loss, sect_loss, sent_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, topic_tr_loss, lm_tr_loss, sect_loss, sent_loss, model, trial, epoch, ignore_keys_for_eval)


            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(self, tr_loss, topic_tr_loss, lm_tr_loss, sect_loss, sent_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}
            # import pdb;
            # pdb.set_trace()
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            sect_loss_scalar = self._nested_gather(sect_loss).mean().item()
            sent_loss_scalar = self._nested_gather(sent_loss).mean().item()
            tp_tr_loss_scalar = self._nested_gather(topic_tr_loss).mean().item()
            lm_tr_loss_scalar = self._nested_gather(lm_tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            sect_loss -= sect_loss
            sent_loss -= sent_loss
            topic_tr_loss -= topic_tr_loss
            lm_tr_loss -= lm_tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            # if self.is_topic_task():
            # logs["topic_loss"] = round(tp_tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            # if self.is_all_task():
            logs["lm_loss"] = round(lm_tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            # if self.is_ext_task():
            logs["sent_loss"] = round(sent_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 6)
            # logs["sect_loss"] = round(sect_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 6)

            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar + \
                                        (sect_loss_scalar + sent_loss_scalar)+ \
                                        tp_tr_loss_scalar + \
                                        lm_tr_loss_scalar


            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        # first save then eval
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
            tr_loss_step, topic_loss_step, lm_loss_step, sect_loss_step, sent_loss_step
        """
        model.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, sent_loss = self.compute_loss(model, inputs)
        lm_loss = loss

        if sent_loss is not None:
            loss = (0.8 * loss) + (0.2 * sent_loss)
        else:
            loss = 0.8 * loss
        ## combine losses
        # import pdb;
        # pdb.set_trace()


        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # topic_loss = topic_loss.mean()   # mean() to average on multi-gpu parallel training
            # sect_loss = sect_loss.mean()   # mean() to average on multi-gpu parallel training
            sent_loss = sent_loss.mean()   # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            # topic_loss = topic_loss / self.args.gradient_accumulation_steps
            # sect_loss = sect_loss / self.args.gradient_accumulation_steps
            if sent_loss is not None:
                sent_loss = sent_loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            # if self.state.global_step==398 or :
            #     import pdb;pdb.set_trace()
            loss.backward()

        # try:
            # sent_loss.detach()
        # except:
        #     import pdb;pdb.set_trace()

        return loss.detach(), \
               lm_loss.detach(), \
               sent_loss.detach() if sent_loss is not None else torch.Tensor([0])[0].cuda()
               # sect_loss.detach(), \
               # topic_loss.detach(), \

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
            if n in self.loading_info['missing_keys']:
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
             'weight_decay': self.args.weight_decay, 'lr': 1e-3},
            {'params': [p for n, p in large_lr_params.items() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': 1e-3},

        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer = OSS(
                params=optimizer_grouped_parameters,
                optim=optimizer_cls,
                **optimizer_kwargs,
            )
        else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


        return self.optimizer


    def sent_ext_step(self, inputs):

        # prepare encoder outputs
        with torch.no_grad():
            return self.model(**inputs)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        sent_extractive=True,
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
        inputs = self._prepare_inputs(inputs)

        # should return only sentence loss
        section_idx = ((inputs['input_ids'][0] == 0).nonzero(as_tuple=True)[0])

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
        with torch.no_grad():
            generated_tokens = self.model.generate(
                generation_inputs,
                ext_labels=inputs['ext_labels'],
                # section_scores=inputs['section_scores'],
                # section_len=inputs['section_len'],
                doc_ids=inputs['doc_ids'],
                section_token_index=section_idx,
                **gen_kwargs,
            )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        if has_labels:
            # labels = inputs["labels"][0][0][None, None, :]
            labels = []
            for batch_labels in inputs["labels"]:
                inside_labels = []
                for label_ids in batch_labels:
                    inside_labels.append(self._pad_tensors_to_max_len(label_ids[None, :], gen_kwargs["max_length"]))

                labels.append(torch.cat(inside_labels, dim=0))

        else:
            labels = None

        return (None, generated_tokens, labels)

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
            inputs_host = None

            # losses/preds/labels on CPU (final containers)
            all_losses = None
            all_preds = None
            all_labels = None
            all_inputs = None

            # Will be useful when we have an iterable dataset so don't know its length.

            observed_num_examples = 0
            # Main evaluation loop
            for step, inputs in enumerate(dataloader):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                        batch_size = observed_batch_size

                # Prediction step
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only,
                                                            ignore_keys=ignore_keys)
                inputs_decode = inputs["input_ids"] if args.include_inputs_for_metrics else None

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
                    labels_host = labels if labels_host is None else labels_host + labels

                if inputs_decode is not None:
                    inputs_decode = self._pad_across_processes(inputs_decode)
                    inputs_decode = self._nested_gather(inputs_decode)
                    inputs_host = (
                        inputs_decode
                        if inputs_host is None
                        else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                    )

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
                        all_preds = logits if all_preds is None else nested_concat(all_preds, logits,
                                                                                   padding_index=-100)
                    if inputs_host is not None:
                        inputs_decode = nested_numpify(inputs_host)
                        all_inputs = (
                            inputs_decode
                            if all_inputs is None
                            else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                        )

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

            if inputs_host is not None:
                inputs_decode = nested_numpify(inputs_host)
                all_inputs = (
                    inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode,
                                                                           padding_index=-100)
                )
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
            if all_inputs is not None:
                all_inputs = nested_truncate(all_inputs, num_samples)

            # Metrics!
            if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
                # if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, doc_ids=ids_host))
                # else:
                #     metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels, doc_ids=ids_host))

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