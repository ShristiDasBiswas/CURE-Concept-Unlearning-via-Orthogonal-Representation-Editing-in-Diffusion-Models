# CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models

As Text-to-Image models continue to evolve, so does the risk of generating unsafe, copyrighted, or privacy-violating content. Existing safety interventions - ranging from training data curation and model fine-tuning to inference-time filtering and guidance - often suffer from incomplete concept removal, susceptibility to jail-breaking, computational inefficiency, or collateral damage to unrelated capabilities. In this paper, we introduce CURE, a training-free concept unlearning framework that operates directly in the weight space of pre-trained diffusion models, enabling fast, interpretable, and highly specific suppression of undesired concepts. At the core of our method is the Spectral Eraser, a closed-form, orthogonal projection module that identifies discriminative subspaces using Singular Value Decomposition over token embeddings associated with the concepts to forget and retain. Intuitively, the Spectral Eraser identifies  and isolates features unique to the undesired concept while preserving safe attributes. This operator is then applied in a single step update to yield an edited model in which the target concept is effectively unlearned - without retraining, supervision, or iterative optimization. To balance the trade-off between filtering toxicity and preserving unrelated concepts, we further introduce an Expansion Mechanism for spectral regularization which selectively modulates singular vectors based on their relative significance to control the strength of forgetting. All the processes above are in closed-form, guaranteeing extremely efficient erasure in only $2$ seconds. Benchmarking against prior approaches, CURE achieves a more efficient and thorough removal for targeted artistic styles, objects, identities, or explicit content, with minor damage to original generation ability and demonstrates enhanced robustness against red-teaming.

## Installation Guide
The code base is based on the `diffusers` package. To get started:
```
conda env create -f environment.yml
conda activate cure
```

## Unlearning
Full unlearning code will be released after publication. For now, we provide model weights post unlearning for a few sample concepts in the `models` folder for users to assess results.

## Model Weights
Sample pre-unlearned model weights are available at:  
📁 [Google Drive – Sample Weights](https://drive.google.com/drive/folders/1HCFa2APFPsJbtq8uBf-dUMge-H-1G2Xn?usp=drive_link)

Place the downloaded `.pt` files into your local `models/` directory:

## Generating Images
To use `generate_images.py` you would need a CSV file with columns `prompt`, `evaluation_seed`, and `case_number`. (Sample data given in `data` folder)
```
python generate_images.py --model_name='models/erased-kelly mckernan.pt' --prompts_path 'data/prompts_erased_kelly.csv' --save_path 'evaluation_folder' --ddim_steps 50
```


## Citation
If you use this work in your research, please cite:

```bibtex
@article{biswas2025cure,
  title   = {CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models},
  author  = {Biswas, Shristi Das and Roy, Arani and Roy, Kaushik},
  journal = {arXiv preprint arXiv:2505.12677},
  year    = {2025}
}
