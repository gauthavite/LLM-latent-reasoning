import torch

class Config:
    modality = "coconut" # "coconut" or "cot" or "no-cot"
    name = "prosqa" # "gsm" or "prosqa"
    save_dir = "."
    model = "gpt2" # "gemma" or "gpt2"

    c_thought = 1
    epochs_per_stage = 5
    max_latent_stage = 6

    batch_size = 8
    num_epochs = 25
    lr = 1e-4
    weight_decay = 0.01

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug = False