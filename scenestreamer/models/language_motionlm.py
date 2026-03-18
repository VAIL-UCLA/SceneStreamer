import logging

import numpy as np
import torch
import torch.nn as nn

from scenestreamer.models.motion_decoder import MotionDecoder
from scenestreamer.models.scene_encoder import SceneEncoder
from scenestreamer.tokenization import get_tokenizer
from scenestreamer.utils import calculate_trajectory_probabilities

logger = logging.getLogger(__file__)


def nucleus_sampling(logits, p=None, epsilon=1e-8):
    p = p or 0.9

    # logits = logits.clamp(-20, 20)

    # Replace NaN and Inf values in logits to avoid errors in entropy computation
    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits).fill_(-1e9), logits)
    logits = torch.where(torch.isinf(logits), torch.zeros_like(logits).fill_(-1e9), logits)

    # Adding a small epsilon to logits to avoid log(0)
    # logits = logits + epsilon

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Sort the probabilities to identify the top-p cutoff
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold p
    cutoff_index = cumulative_probs > p
    # Shift the mask to the right to keep the first token above the threshold
    cutoff_index[..., 1:] = cutoff_index[..., :-1].clone()
    cutoff_index[..., 0] = False

    # Zero out the probabilities for tokens not in the top-p set
    sorted_probs.masked_fill_(cutoff_index, 0)

    # Recover the original order of the probabilities
    original_probs = torch.zeros_like(probs)
    original_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

    # original_probs += epsilon

    # Sample from the adjusted probability distribution
    # try:
    sampled_token_index = torch.distributions.Categorical(probs=original_probs).sample()
    # except ValueError:
    #     import ipdb; ipdb.set_trace()
    #     print(1111111)

    return sampled_token_index


class LanguageMotionLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scene_encoder = SceneEncoder(config=self.config)
        self.motion_decoder = MotionDecoder(config=self.config)

        from transformers import BertModel, BertTokenizer

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if not config.FINE_TUNE_BERT:  # if not finetuning BERT encoder for our prompt
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.bert_projection = nn.Linear(self.bert_model.config.hidden_size, 512)

    def encode_prompt(self, prompts):
        # Tokenize the batch of prompts
        encoded_input = self.bert_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

        # Ensure the input tensors are on the same device as the model
        device = next(self.bert_model.parameters()).device
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        # Encode the batch of prompts using BERT
        with torch.no_grad():
            bert_output = self.bert_model(**encoded_input)

        # Use the output of the [CLS] token for each prompt in the batch
        cls_embedding = bert_output.last_hidden_state[:, 0, :]

        # Project BERT embedding to the desired dimensionality
        prompt_embedding = self.bert_projection(cls_embedding)

        return prompt_embedding

    def forward(self, input_dict):
        print("in scene encoder, input dict:", input_dict.key())

        # if self.config.LANGUAGE_CONDITION and 'decoder/text_label' in input_dict:
        #     prompt = input_dict['decoder/text_label']
        #     print("text_label:", text_label)
        #     prompt_embedding = self.encode_prompt(prompt)
        #     print("prompt embedding:", prompt_embedding)
        #     input_dict['decoder/prompt_embedding'] = prompt_embedding

        # else:
        #     print("NOOOOO")
        #     print(input_dict.keys())
        #     print(self.config.LANGUAGE_CONDITION)

        input_dict = self.encode_scene(input_dict)
        input_dict = self.decode_motion(input_dict)
        return input_dict

    def encode_scene(self, input_dict):
        return self.scene_encoder(input_dict)

    def decode_motion(self, input_dict, use_cache=False):
        input_dict = self.motion_decoder(input_dict, use_cache=use_cache)
        return input_dict

    def autoregressive_rollout(
        self,
        input_dict,
        num_decode_steps,
        num_prev_steps=1,
        use_cache=True,
        sampling_method="softmax",
        temperature=None,
        topp=None,
        num_modes_for_eval=None
    ):
        if temperature is None:
            temperature = self.config.SAMPLING.TEMPERATURE
        if topp is None:
            topp = self.config.SAMPLING.TOPP

        B, T_input, N = input_dict["decoder/input_action"].shape
        assert num_decode_steps >= 1
        assert input_dict["decoder/input_action_valid_mask"].shape == (B, T_input, N)
        assert T_input >= num_prev_steps

        # Record "current" valid mask of input actions, we'll repeat it for each decoding step.
        input_action_valid_mask = torch.clone(
            input_dict["decoder/input_action_valid_mask"][:, num_prev_steps - 1:num_prev_steps]
        )

        # Discard future actions / mask
        input_dict["decoder/input_action"] = input_dict["decoder/input_action"][:, :num_prev_steps]
        input_dict["decoder/input_action_valid_mask"] = \
            input_dict["decoder/input_action_valid_mask"][:, :num_prev_steps]

        if self.config.MODEL.RELATIVE_PE_DECODER:
            input_dict["decoder/modeled_agent_heading"] = input_dict["decoder/modeled_agent_heading"][:, :num_prev_steps
                                                                                                      ]
            input_dict["decoder/modeled_agent_position"] = input_dict["decoder/modeled_agent_position"
                                                                      ][:, :num_prev_steps]

            tokenizer = get_tokenizer(config=self.config)

            original_data = {
                "decoder/current_agent_position": input_dict["decoder/current_agent_position"].clone(),
                "decoder/current_agent_heading": input_dict["decoder/current_agent_heading"].clone(),
                "decoder/current_agent_velocity": input_dict["decoder/current_agent_velocity"].clone(),
            }

        # Get scene embedding
        input_step = torch.arange(num_decode_steps).to(input_dict["encoder/agent_position"].device)

        # ================ for language labels condition
        try:
            prompts = input_dict['decoder/text_label']
            print("text_label:", prompts)

            if isinstance(prompts, np.ndarray):
                prompts = prompts.tolist()

            prompt_embedding = self.encode_prompt(prompts)
            print("prompt embedding:", prompt_embedding)
            input_dict['decoder/prompt_embedding'] = prompt_embedding

        except Exception as e:
            import pdb
            pdb.set_trace()
            exit()

        # ===============

        input_dict = self.encode_scene(input_dict)
        output_logit_list = []
        output_action_list = []
        input_dict["decoder/input_step"] = input_step[:1]
        for decode_step in range(num_decode_steps):
            logger.debug(f"======================= STEP {decode_step=} =======================")

            if not use_cache:
                input_dict["decoder/input_step"] = input_step[:decode_step + 1]

            # Decode motion tokens
            input_dict = self.decode_motion(input_dict, use_cache=use_cache)

            output_token = input_dict["decoder/output_logit"]

            if use_cache:
                assert output_token.shape[:3] == (B, 1, N)
            else:
                assert output_token.shape[:3] == (B, decode_step + 1, N)
                output_token = output_token[:, -1:]  # -> output_token.shape == (B, 1, N, #actions)

            output_logit_list.append(output_token)

            # Sample the action
            if sampling_method == "argmax":
                selected_action = output_token.argmax(-1)
            elif sampling_method == "softmax":
                selected_action = torch.distributions.Categorical(logits=output_token / temperature).sample()
            elif sampling_method == "topp":
                selected_action = nucleus_sampling(logits=output_token / temperature, p=topp)
            else:
                raise ValueError("Unknown sampling method: {}".format(sampling_method))

            output_action_list.append(selected_action)

            if use_cache:
                # Discard the previous tokens whose key/value are cached.
                input_dict["decoder/input_action"] = selected_action
                input_dict["decoder/input_action_valid_mask"] = input_action_valid_mask
                input_dict["decoder/input_step"].fill_(decode_step + 1)

                if self.config.MODEL.RELATIVE_PE_DECODER:
                    reconstructed_pos, reconstructed_heading, reconstructed_vel = tokenizer.detokenize_for_step(
                        data_dict=input_dict,
                        action=selected_action,
                    )
                    input_dict["decoder/current_agent_position"] = reconstructed_pos
                    input_dict["decoder/current_agent_heading"] = reconstructed_heading
                    input_dict["decoder/current_agent_velocity"] = reconstructed_vel
                    input_dict["decoder/modeled_agent_heading"] = reconstructed_heading.reshape(B, 1, N)
                    input_dict["decoder/modeled_agent_position"] = reconstructed_pos.reshape(B, 1, N, 2)

            else:
                input_dict["decoder/input_action"] = torch.cat(
                    [input_dict["decoder/input_action"], selected_action], dim=1
                )
                input_dict["decoder/input_action_valid_mask"] = torch.cat(
                    [input_dict["decoder/input_action_valid_mask"], input_action_valid_mask], dim=1
                )

            assert input_dict["decoder/input_action"].shape == input_dict["decoder/input_action_valid_mask"].shape

        output_action_list = torch.concatenate(output_action_list, dim=1)
        assert output_action_list.shape == (B, num_decode_steps, N)

        output_logit_list = torch.concatenate(output_logit_list, dim=1)
        input_dict["decoder/output_logit"] = output_logit_list
        input_dict["decoder/output_action"] = output_action_list
        input_dict["decoder/output_score"] = calculate_trajectory_probabilities(
            output_logit_list, output_action_list, mask=input_action_valid_mask
        )  # (B, N)

        if self.config.MODEL.RELATIVE_PE_DECODER:
            input_dict["decoder/current_agent_position"] = original_data["decoder/current_agent_position"]
            input_dict["decoder/current_agent_heading"] = original_data["decoder/current_agent_heading"]
            input_dict["decoder/current_agent_velocity"] = original_data["decoder/current_agent_velocity"]

        return input_dict
