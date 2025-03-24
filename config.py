import torch

class Config:
    modality = "coconut" # "cot"
    name = "prosqa"
    save_dir = f"/Data/checkpoints/test"

    c_thought = 1
    epochs_per_stage = 5
    max_latent_stage = 6

    batch_size = 8
    num_epochs = 50
    lr = 1e-4
    weight_decay = 0.01

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug = False