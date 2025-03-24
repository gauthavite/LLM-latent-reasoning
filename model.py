import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

class Coconut(nn.Module):
    def __init__(
        self,
        base_causallm,
        latent_token_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids):
        logits = []

        latent_indices = (input_ids == self.latent_token_id).nonzero()

        # For each batch instance, collect the positions of its latent tokens
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max(len(l) for l in latent_lists)

        # Initial range for computing forward
        if max_n_latents > 0:
            earliest_latent = latent_indices[:, 1].min().item()
            next_compute_range = (0, earliest_latent)
        else:
            next_compute_range = (0, input_ids.shape[1])

        inputs_embeds = self.embedding(input_ids)

        for pass_idx in range(max_n_latents):
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                output_hidden_states=True
            )
            logits.append(outputs.logits)

            hidden_states = outputs.hidden_states[-1]

            # We will now update the embeddings with the hidden states for the latent tokens.
            filling_indices = [(inst_i, lat_positions[pass_idx]) for inst_i, lat_positions in enumerate(latent_lists) if len(lat_positions) > pass_idx]

            # We'll do updates on a python list-of-lists first to avoid in-place on tensor
            tensor_list = [[inputs_embeds[b_i, seq_pos, :] for seq_pos in range(inputs_embeds.shape[1])] for b_i in range(inputs_embeds.shape[0])]

            for (batch_idx, token_idx) in filling_indices:
                tensor_list[batch_idx][token_idx] = hidden_states[batch_idx, token_idx - 1 - next_compute_range[0], :]

            # Reassemble into a single tensor
            inputs_embeds = torch.stack(
                [torch.stack(tensor_list[b_i]) for b_i in range(inputs_embeds.shape[0])]
            )

            start = next_compute_range[1]
            if pass_idx + 1 >= max_n_latents:
                end = input_ids.shape[1]
            else:
                end = next_compute_range[1] + 1
            next_compute_range = (start, end)


        outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                output_hidden_states=True,
            )
        logits.append(outputs.logits)

        logits = torch.cat(logits, dim=-2)

        # Standard language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(self, input_ids, max_new_tokens=16):

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs = self.forward(input_ids, attention_mask, labels, position_ids)

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        next_token_embed = self.embedding(torch.tensor([[next_token]], device=input_ids.device))
        inputs_embeds = torch.cat((outputs.inputs_embeds, next_token_embed), dim=1)

        for _ in range(max_new_tokens):
            outputs = self.base_causallm(inputs_embeds=inputs_embeds)
            next_token = torch.argmax(outputs.logits[0, -1]).item()

            if next_token == self.eos_token_id:
                break

            tokens.append(next_token)

            next_token_embed = self.embedding(
                torch.tensor([[next_token]], device=input_ids.device)
            )
            inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)

        return torch.tensor(tokens, device=input_ids.device).unsqueeze(0)