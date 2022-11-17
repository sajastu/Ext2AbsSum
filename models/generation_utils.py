import inspect
import warnings
from typing import Optional, Iterable, Callable, List, Union, Dict, Any, Tuple

from torch import nn
import logging

from transformers import LogitsProcessorList, StoppingCriteriaList, Constraint, ConstrainedBeamSearchScorer, \
    BeamScorer, BeamSearchScorer
from transformers.file_utils import ModelOutput
from transformers.generation_stopping_criteria import validate_stopping_criteria
from transformers.generation_utils import GenerationMixin, GreedySearchOutput, SampleOutput, BeamSearchOutput, \
    BeamSampleOutput, BeamSearchDecoderOnlyOutput, BeamSearchEncoderDecoderOutput
import torch

from transformers.pytorch_utils import torch_int_div
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

class GenerationMixin(GenerationMixin):

    def get_repr_from_index_gen(self, from_tensor, index):
        return from_tensor[:,index, :]

    def _prepare_reduced_encoder_outputs(self, encoder_outputs, input_ids, section_len):

        # if input_ids[1]
        sent_repr = self.get_repr_from_index_gen(encoder_outputs[0],
                                             index=((input_ids[0] == 0).nonzero(as_tuple=True)[0]))

        section_repr = self.get_repr_from_index_gen(encoder_outputs[0],
                                                index=((input_ids[0] == input_ids[0][0]).nonzero(as_tuple=True)[0]))

        # import pdb;pdb.set_trace()
        sent_scorer_ln = self.get_sent_scorer()
        sent_scores = torch.sigmoid(
            sent_scorer_ln(sent_repr)
        ).squeeze(-1)


        sect_scorer_ln = self.get_sect_scorer()
        sect_scores = torch.nn.functional.softmax(sect_scorer_ln(section_repr),
                                                  dim=-2).squeeze(-1)

        LIMIT = 4096  # tokens

        if self.SAMPLING_FROM=='section':
            sample_sect_dist = torch.round(
                torch.tensor([LIMIT] * (section_repr.size(1))).unsqueeze(0).cuda() * sect_scores.squeeze(-1))
            sent_real_ids = (input_ids[0] == 0).nonzero(as_tuple=True)[0]
            end_pre_ids = (input_ids[0] == 2).nonzero(as_tuple=True)[0]

            sent_len = ((end_pre_ids - sent_real_ids) + 1)[None, :]
            sects_batch_sent_scores = pad_sequence(torch.split(sent_scores[0], section_len[0].tolist()), batch_first=True,
                                                   padding_value=-1)[None, :, :].repeat(input_ids.size(0), 1, 1)

            sects_batch_sent_lens = pad_sequence(torch.split(sent_len[0], section_len[0].tolist()), batch_first=True,
                                                 padding_value=0)[None, :, :].repeat(input_ids.size(0), 1, 1)
            sect_sent_mask = (sects_batch_sent_lens > 0).float()

            top_sents_idxs = torch.argsort(sects_batch_sent_scores, descending=True)
            # test
            top_sects_batch_sent_lens = torch.zeros_like(top_sents_idxs).cuda()
            top_sects_batch_sent_lens.scatter_(2, top_sents_idxs, sects_batch_sent_lens)

            sect_sent_n_selects = (~(
                        (torch.cumsum(top_sects_batch_sent_lens, dim=-1)) > sample_sect_dist[:, :, None].expand_as(
                    top_sects_batch_sent_lens)) * sect_sent_mask).sum(dim=-1)
            top_sents_mask = ((pad_sequence([torch.arange(xx) for x in sect_sent_n_selects for xx in x],
                                            padding_value=-1).t()) != -1).cuda()
            pre_idx = top_sents_idxs[:, :, :int(sect_sent_n_selects.max().item())] * top_sents_mask

            section_len_cum = torch.cumsum(section_len, dim=-1)
            shifted_section_len = torch.cat((torch.tensor([0])[None, :].cuda(), section_len_cum), dim=-1)
            # pre_idx_hyp = torch.cat((pre_idx, torch.zeros_like(pre_idx)[:, :, :1]), dim=-1)
            shifted_section_len = shifted_section_len[:, :, None].repeat(1, 1, pre_idx.size(2))[:, :-1, :]
            pre_idx = (pre_idx + shifted_section_len) * top_sents_mask

            pre_idx = torch.where(pre_idx > 0, sent_real_ids[pre_idx], 10000000)
            end_pre_ids = (pre_idx + top_sects_batch_sent_lens[:, :, :pre_idx.size(-1)])

            # sort ids
            pre_idx = pre_idx.sort()[0]
            end_pre_ids = end_pre_ids.sort()[0]

            sections_sentence_encoding = []
            selected_sent_embeddings = []

            for l in range(pre_idx.size(1)):
                # import pdb;
                # pdb.set_trace()

                section_sent_encoding = []
                start_idxs = pre_idx[0, l]
                end_idxs = end_pre_ids[0, l]

                if top_sents_mask[l].float().sum() == 0:
                    continue

                for ll in range(start_idxs.size(-1)):
                    # import pdb;pdb.set_trace()
                    if start_idxs[ll] > 10000000 - 1:
                        continue
                    try:
                        sent_encoding = encoder_outputs[0][:, start_idxs[ll]:end_idxs[ll] + 1]
                    except:
                        sent_encoding = encoder_outputs[0][:, start_idxs[ll]:end_idxs[ll]]

                    selected_sent_embeddings.append(encoder_outputs[0][:, start_idxs[ll]].unsqueeze(0))
                    section_sent_encoding.append(sent_encoding)
                try:
                    sections_sentence_encoding.append(torch.cat(section_sent_encoding, dim=1))
                except:
                    import pdb;pdb.set_trace()
            sections_sentence_encoding = torch.cat(sections_sentence_encoding, dim=1).cuda()
            selected_sent_embeddings = torch.cat(selected_sent_embeddings, dim=0).cuda()

            if self.is_hier():
                hier_encoder = self.get_hier_encoder()
                selected_sent_embeddings = hier_encoder(selected_sent_embeddings,
                                                         torch.ones(selected_sent_embeddings.size(0),
                                                                    selected_sent_embeddings.size(1)).bool().cuda())

            return sections_sentence_encoding, selected_sent_embeddings

        else:
            sent_real_ids = (input_ids[0] == 0).nonzero(as_tuple=True)[0]
            end_pre_ids = (input_ids[0] == 2).nonzero(as_tuple=True)[0]

            sent_len = ((end_pre_ids - sent_real_ids) + 1)[None, :]

            top_sents_ids = torch.argsort(sent_scores, descending=True)
            top_sents_len = torch.zeros_like(sent_len)

            # top_sents_len.scatter_(1, top_sents_ids, sent_len)
            top_sents_len = torch.index_select(sent_len, 1, top_sents_ids[0])

            top_sents_included = (~(torch.cumsum(top_sents_len, dim=-1) > LIMIT)).sum()
            top_sents_ids = top_sents_ids[:, :top_sents_included]

            top_sents_start_ids = sent_real_ids[top_sents_ids.sort(dim=1)[0]]
            sent_len = torch.index_select(sent_len, 1, top_sents_ids.sort(dim=1)[0].squeeze(0))
            top_sens_end_ids = top_sents_start_ids + sent_len

            masked_top_sents_input = torch.zeros_like(input_ids)
            num_of_masked = 0
            selected_sents_encodings = []
            for start_idx, end_idx in zip(top_sents_start_ids[0], top_sens_end_ids[0]):
                masked_top_sents_input[:, start_idx:end_idx] = 1
                num_of_masked += len(masked_top_sents_input[0, start_idx:end_idx])
                selected_sents_encodings.append(encoder_outputs[0][:,start_idx])
            # import pdb;pdb.set_trace()

            selected_sents_encodings = torch.cat(selected_sents_encodings, dim=0).cuda()
            # torch.where(masked_top_sents_input > -1, input_ids,)
            mask = masked_top_sents_input.unsqueeze(-1).expand_as(encoder_outputs[0]).bool()
            reduced_encoder_outputs = torch.masked_select(encoder_outputs[0], mask).view(1, num_of_masked, -1)
            if self.is_hier():
                hier_encoder = self.get_hier_encoder()
                selected_sents_encodings = selected_sents_encodings.unsqueeze(0)
                selected_sents_encodings = hier_encoder(selected_sents_encodings,
                                                         torch.ones(selected_sents_encodings.size(0),
                                                                    selected_sents_encodings.size(1)).bool().cuda())

            return reduced_encoder_outputs, selected_sents_encodings


    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, section_token_index=None, model_kwargs=None, model_input_name: Optional[str]=None, first=False,
    ) -> Dict[str, Any]:
        # 1. get encoder

        if "encoder_outputs" not in model_kwargs:

            encoder = self.get_encoder()
            # 2. prepare encoder args and encoder kwargs from model kwargs
            irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "src_bow_global", "ext_labels", "section_token_index", "section_scores", "section_len"]

            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not any(argument.startswith(p) for p in irrelevant_prefix)
            }

            # 3. make sure that encoder returns `ModelOutput`
            model_input_name = model_input_name if model_input_name is not None else self.main_input_name
            encoder_kwargs["return_dict"] = True
            encoder_kwargs[model_input_name] = inputs_tensor
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        if first:
            # only pick out abstract and intro
            sect_idx = ((inputs_tensor[0] == 50265).nonzero(as_tuple=True)[0])
            try:
                model_kwargs["reduced_encodings"] = model_kwargs["encoder_outputs"][0][0][:sect_idx[2]].unsqueeze(0)
            except:
                # the instance has two sections
                model_kwargs["reduced_encodings"] = model_kwargs["encoder_outputs"][0]

            model_kwargs["reduced_encodings_mask"] = torch.ones(model_kwargs["reduced_encodings"].shape[0], model_kwargs["reduced_encodings"].shape[1]).cuda()

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: Optional[torch.LongTensor] = None,
            reduced_encodings_mask: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[ModelOutput] = None,
            reduced_encodings: Optional[ModelOutput] = None,
            encoder_outputs_section: Optional[ModelOutput] = None,
            selected_sent_embeddings: Optional[ModelOutput] = None,
            sections_sentence_encoding: Optional[ModelOutput] = None,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor,  Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            # encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            #     0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            # )
            model_kwargs["encoder_outputs"] = encoder_outputs[0].index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device))

            model_kwargs["reduced_encodings"] = reduced_encodings.index_select(
                0, expanded_return_idx.to(reduced_encodings_mask.device))
            model_kwargs["reduced_encodings_mask"] = reduced_encodings_mask.index_select(
                0, expanded_return_idx.to(reduced_encodings_mask.device))

            # model_kwargs["selected_sent_embeddings"] = selected_sent_embeddings.index_select(
            #     0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device))

            # model_kwargs["sections_sentence_encoding"] = sections_sentence_encoding.index_select(
            #     0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device))

        return input_ids, model_kwargs

    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            src_bow_global=None,
            doc_ids=None,
            ext_labels=None,
            section_scores=None,
            section_token_index=None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            typical_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            max_time: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
            renormalize_logits: Optional[bool] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
            constraints: Optional[List[Constraint]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            forced_bos_token_id: Optional[int] = None,
            forced_eos_token_id: Optional[int] = None,
            remove_invalid_values: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
            **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:

        # sent_boundaries = ((inputs[0] == 0).nonzero(as_tuple=True)[0], (inputs[0] == 2).nonzero(as_tuple=True)[0])
        start_ids = (inputs[0] == 0).nonzero(as_tuple=True)[0]
        end_ids = torch.cat((start_ids[1:], torch.Tensor([inputs.shape[-1] - 2]).cuda()), dim=-1).int()
        sent_boundaries = (start_ids, end_ids)
        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            # special case if pad_token_id is not defined
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache
        model_kwargs["ext_labels"] = ext_labels
        # model_kwargs["section_scores"] = section_scores
        # model_kwargs["doc_ids"] = doc_ids

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`

            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, section_token_index, model_kwargs, model_input_name, first=True)


        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids.shape[-1]
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # 6. determine generation mode
        num_beams = 2
        no_repeat_ngram_size = 3
        max_length = 256
        min_length = 80
        length_penalty = 1.0
        early_stopping = True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        input_ids_seq_length = input_ids.shape[-1]

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )

        # 9. go into different generation modes
        if is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )

            # 11. interleave input_ids with `num_beams` additional sequences per batch

            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            model_kwargs['src_ids'] = inputs

            return self.beam_search(
                input_ids,
                beam_scorer,
                sent_boundaries=sent_boundaries,
                section_token_index=section_token_index,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )


    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        sent_boundaries=None,
        section_token_index= None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = True,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only


        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            # import pdb;pdb.set_trace()

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # import pdb;
            # pdb.set_trace()

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required


            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless

            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            # import pdb;pdb.set_trace()
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            nxt_token = beam_next_tokens.unsqueeze(-1)
            # mask_swch = (nxt_token == 50267)
            input_ids = torch.cat([input_ids[beam_idx, :], nxt_token], dim=-1)
            last_token = input_ids[:, -1]
            mask_swch = (last_token == 50267)

            if mask_swch.sum() > 0:
                # print(cur_len)
                # EOS is detected in some indices
                # should update 'reduced_encodings' keys of the detected indices
                swch_indx = ((mask_swch.squeeze(-1) == True).nonzero(as_tuple=True)[0])
                # nonSwth__indx = ((mask_swch.squeeze(-1) == False).nonzero(as_tuple=True)[0])
                # switch_partial_encodings = outputs.decoder_last_hidden_states[swch_indx, :, :]
                input_ids_swch = input_ids[swch_indx, :]

                switch_encoder_outputs = model_kwargs['encoder_outputs'][swch_indx, :, :]
                # nonSwitch_encoder_outputs = model_kwargs['reduced_encodings'][nonSwth__indx, :, :]


                encoder = self.get_encoder()
                input_ids_swch_x = input_ids_swch
                input_ids_swch_x[:, 0] = 0
                input_ids_swch_x[input_ids_swch_x==50267] = 0
                summary_attention_mask = torch.zeros_like(input_ids_swch_x).cuda()
                summary_attention_mask[input_ids_swch_x!=1] = 1
                global_attention_mask = torch.zeros_like(summary_attention_mask)
                global_attention_mask[input_ids_swch_x==0] = 1
                summary_encoder_outputs = encoder(input_ids=input_ids_swch_x, attention_mask=summary_attention_mask, global_attention_mask=global_attention_mask)
                combiner_mask = summary_attention_mask
                combiner_mask = combiner_mask[:, None, None, :].repeat(1, 1, model_kwargs['encoder_outputs'].shape[1], 1)

                # updating reduced_encodings of the changed indices...
                combiner = self.get_combiner()
                encoder_outputs_x, cross_attn_weights_word, cross_attn_present_key_value_word = combiner(hidden_states=switch_encoder_outputs, key_value_states=summary_encoder_outputs[0], attention_mask=combiner_mask)
                extractor = self.get_extractor()
                updated_rows_reduced_encodings, updated_rows_reduced_encodings_mask, updated_rows_sent_scores = extractor(encoder_outputs_x, sent_boundaries, LIMIT=1024)

                bs_old, seq_len_old, dim = model_kwargs['reduced_encodings'].shape
                bs_updated, seq_len_updated, _ = updated_rows_reduced_encodings.shape
                if seq_len_updated > seq_len_old:
                    # pad the old reduced encodings and put updated_rows instead.
                    model_kwargs['reduced_encodings'] = torch.cat((model_kwargs['reduced_encodings'], torch.ones(bs_old, seq_len_updated-seq_len_old, dim).cuda()), dim=1)
                    model_kwargs['reduced_encodings_mask'] = torch.cat((model_kwargs['reduced_encodings_mask'], torch.zeros(bs_old, seq_len_updated-seq_len_old).cuda()), dim=1)
                    # cid = 0
                    # for indx_updated in swch_indx:
                    #     model_kwargs['reduced_encodings'][indx_updated] = updated_rows_reduced_encodings[cid]
                    #     model_kwargs['reduced_encodings_mask'][indx_updated] = updated_rows_reduced_encodings_mask[cid]
                    #     cid +=1

                else:
                    # pad the updated_rows and then put into the old reduced_encod
                    updated_rows_reduced_encodings = torch.cat((updated_rows_reduced_encodings, torch.ones(bs_updated, seq_len_old-seq_len_updated, dim).cuda()), dim=1)
                    updated_rows_reduced_encodings_mask = torch.cat((updated_rows_reduced_encodings_mask, torch.zeros(bs_updated, seq_len_old-seq_len_updated).cuda()), dim=1)

                # now replacing
                cid = 0
                for indx_updated in swch_indx:
                    model_kwargs['reduced_encodings'][indx_updated] = updated_rows_reduced_encodings[cid]
                    model_kwargs['reduced_encodings_mask'][indx_updated] = updated_rows_reduced_encodings_mask[cid]
                    cid += 1



            # mask_swch_row, mask_swch_col = torch.where(input_ids)
            # input_ids[input_ids==50267]
            # (50267 in input_ids)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True


        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        # TODO
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('/disk1/sajad/sci-trained-models/bart/mentsum/checkpoint-50000/')
        # tokenizer = AutoTokenizer.from_pretrained('/disk0/shabnam/.cache/sci-trained-models/grease/mentsum-grease-encoder6-node750-withAttn-v5/')
        # src_ids = model_kwargs['src_ids'].cpu().numpy()
        # sequence = sequence_outputs['sequences'].cpu().numpy()
        # sequences_attn = sequence_outputs['sequences_attn']
        # mask = model_kwargs['attention_mask']
        # for beam_idx in range(mask.shape[0]):
        #     truncated_idx = mask.sum(dim=1).cpu().numpy()
        # attn_idx = 0
        # self.plot_attention(src_ids[0][:truncated_idx[attn_idx]], sequence[0][:truncated_idx[attn_idx]], sequences_attn, tokenizer,
        #                     f'analysis/{model_kwargs["doc_ids"][0]}-grease')

        # import pdb;pdb.set_trace()
        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            # import pdb;
            # pdb.set_trace()
            return sequence_outputs["sequences"]