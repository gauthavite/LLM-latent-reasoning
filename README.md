# LLM-latent-reasoning
This repository provides a simplified and easy-to-understand implementation of **Coconut**, inspired by the paper:

> **"Training Large Language Models to Reason in a Continuous Latent Space"**  
> [Paper Link](https://arxiv.org/abs/2310.02089) | [Official Repository](https://github.com/facebookresearch/coconut)

## How to run 
1.	Update the configuration in ```config.py```.


2.	Place your data in the data directory.

3.	Launch training: ```python main.py```.

4.	Monitor checkpoints in the ```Config.save_dir``` folder. The best model based on validation accuracy is saved as ```best_model.pt```.

## Acknowledgments

This implementation was inspired by the original [Coconut](https://github.com/facebookresearch/coconut) paper and repository. Please refer to the official repo for full details and advanced use cases.
