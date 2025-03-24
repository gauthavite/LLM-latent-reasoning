import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from functools import partial

import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
# from unsloth import FastModel

from model import Coconut
from config import Config
from dataset import CoconutDataset, collate_fn

model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens("<|start-latent|>")
tokenizer.add_tokens("<|end-latent|>")
tokenizer.add_tokens("<|latent|>")
latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

#resize model token embeddings
model.resize_token_embeddings(len(tokenizer))

if Config.modality == "coconut":
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    # initialize the new token embeddings with a known token
    for token_id in [latent_id, start_id, end_id]:
        target_embedding = embeddings.weight.data[target_id]
        embeddings.weight.data[token_id] = target_embedding
    model = Coconut(model, latent_id, tokenizer.eos_token_id)


if not os.path.exists(Config.save_dir):
    os.makedirs(Config.save_dir)

question_val = [
    d["question"] for d in json.load(open(f"data/{Config.name}_valid.json"))
]
answers_val = [
    d["answer"].replace(",", "").strip()
    for d in json.load(open(f"data/{Config.name}_valid.json"))
]
cot_val = [
    "\n".join(d["steps"]) for d in json.load(open(f"data/{Config.name}_valid.json"))
]

model.to(Config.device)

max_new_tokens = 128

optimizer = optim.AdamW(
    model.parameters(),
    lr=Config.lr,
    weight_decay=Config.weight_decay,
)
scaler = torch.amp.GradScaler()

collate = partial(collate_fn, tokenizer=tokenizer, latent_id=latent_id)
best_acc = 0


for epoch in range(Config.num_epochs):
    scheduled_stage = epoch // Config.epochs_per_stage if Config.modality == "coconut" else 0

    dataset_train = CoconutDataset(
        Config.name,
        "train",
        Config.modality,
        scheduled_stage,
        Config.debug,
        tokenizer,
        start_id,
        end_id,
        latent_id,
        Config.c_thought,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=1,
        pin_memory=True,
        batch_size=Config.batch_size,
        collate_fn=collate,
    )

    dataset_val = CoconutDataset(
        Config.name,
        "valid",
        Config.modality,
        scheduled_stage,
        Config.debug,
        tokenizer,
        start_id,
        end_id,
        latent_id,
        Config.c_thought,
    )

    valid_gen_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        num_workers=1,
        pin_memory=True,
        batch_size=1,
        collate_fn=collate,
    )

    model.train()
    train_losses = []
    for batch in tqdm(train_dataloader, desc=f"train epoch {epoch + 1}"):
        batch = {key: batch[key].to(Config.device) for key in batch.keys() if key != "idx"}

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_losses.append(loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses)
    print(
        f"Epoch {epoch + 1}/{Config.num_epochs} - Avg Train Loss: {avg_train_loss:.4f}"
    )

    correct = torch.tensor(0, device=Config.device)
    correct_cot = torch.tensor(0, device=Config.device)
    total = torch.tensor(0, device=Config.device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_gen_dataloader, desc=f"valid epoch {epoch + 1}"):
            test_idx = batch["idx"][0]

            batch = {
                k: v.to(Config.device)
                for k, v in batch.items()
                if v != None and k not in ["idx", "position_ids"]
            }

            answer = answers_val[test_idx.cpu().item()]
            answer_cot = cot_val[test_idx.cpu().item()]
            question = question_val[test_idx.cpu().item()]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model.generate(**batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)

            text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_output = text_output.split("#")[-1].replace(",", "").strip()
            cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()

            correct += answer_output == answer
            correct_cot += cot_output == answer_cot
            total += 1

    test_accuracy = correct / total
    cot_accuracy = correct_cot / total

    print(
        f"Epoch {epoch + 1}/{Config.num_epochs} - Validation accuracy: {test_accuracy:.2f} | CoT validation match: {cot_accuracy:.2f}"
    )

    if test_accuracy > best_acc:
        best_acc = test_accuracy
        checkpoint_path = os.path.join(Config.save_dir, "best_model.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved new best model with accuracy {best_acc:.4f} at {checkpoint_path}")