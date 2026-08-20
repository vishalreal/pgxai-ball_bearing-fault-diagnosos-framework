# PGXAI: A Physics-Grounded Framework for Evaluating Explainable AI in Bearing Fault Diagnosis

MSc Practicum project — School of Computing, Dublin City University.

**Authors:** Vishal Surendran, Venkata Sujith Chaitanya Padavala
**Supervisor:** Dr. Sunder Ali Khowaja

---

## What this project does

Deep learning models for bearing fault diagnosis routinely report near-perfect classification accuracy. But accuracy alone doesn't tell you whether a model's decision is grounded in the real physical cause of the fault, or whether it's a lucky correlation. Explainable AI techniques like Grad-CAM and attention maps are usually shown as illustrative pictures — nobody checks whether the highlighted regions actually correspond to known fault physics.

This project introduces **PGXAI**, an evaluation framework that answers that question quantitatively, and applies it to compare a CNN (ResNet50) and a Vision Transformer (ViT-B16) on the CWRU bearing fault dataset.

## The core question

> Does high classification accuracy in CNN/ViT bearing fault diagnosis imply the model's explanations are physically grounded — or can accuracy and explanatory validity move independently of each other?

## What PGXAI consists of

1. **Leakage-free evaluation protocol** — train and test sets built from physically different recordings (1750 rpm train / 1797 rpm test), avoiding the near-duplicate leakage that occurs when overlapping signal windows are split randomly.
2. **Fault Frequency Alignment Score (FFAS)** — quantifies what fraction of a saliency map's energy falls within the theoretically-derived fault frequency band (BPFI / BPFO / BSF, computed from real bearing geometry), rather than just visualising the heatmap.
3. **Random-heatmap baseline** — 100 meaningless random heatmaps scored the same way, establishing a falsifiable chance-level floor so FFAS scores can't just be declared "good" arbitrarily.
4. **Statistical validation** — one-sample t-tests and Cohen's d effect sizes, per class, per model, per split condition.

## Key findings

- Under a naïve random-segment split, both ResNet50 and ViT-B16 achieve **100% accuracy** — but under the leakage-free, cross-speed split, accuracy drops to **84.90% (ResNet50)** and **84.25% (ViT-B16)**, showing that a large share of the original accuracy was near-duplicate memorisation, not genuine generalisation.
- Under the leakage-free split, mean FFAS deviation from random baseline was **−0.0165 (ResNet50)** and **−0.0176 (ViT-B16)** — both close to, or slightly below, chance level.
- Leakage's effect on FFAS is small and **inconsistent in direction between architectures** (ResNet50: +0.0071 under leakage vs. −0.0165 without; ViT-B16: −0.0187 under leakage vs. −0.0176 without) — ruling out the idea that higher accuracy alone yields stronger physical grounding.
- **Classification accuracy and physics-based explanatory validity are largely independent properties** of these models on this task.

## Bearing fault frequencies used

For the SKF 6205 bearing (n = 9 balls, d = 0.3126", D = 1.537", θ = 0°) at the 1797 rpm test condition:

| Fault type | Frequency |
|---|---|
| BPFO (Outer race) | 107.4 Hz |
| BPFI (Inner race) | 162.2 Hz |
| BSF (Ball) | 141.2 Hz |

## Repository contents

```
pgxai.ipynb      # Full pipeline: preprocessing → training → XAI → FFAS → results
.gitignore       # Excludes large data/model files (see below)
```

**Note:** raw CWRU `.mat` files, generated `.npy` datasets, and trained model checkpoints (`.pth`) are excluded via `.gitignore` due to size. See setup instructions below to regenerate them.

## Setup and reproduction

1. Download the CWRU 12kHz Drive End bearing dataset (Normal + 9 fault classes: IR007/014/021, B007/014/021, OR007/014/021) from the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter), for both the 2HP (1750 rpm) and 0HP (1797 rpm) load conditions.
2. Place the `.mat` files in the folder structure expected by the notebook (see the data-loading cells at the top of `pgxai.ipynb`).
3. Run the notebook top to bottom in Google Colab or a local Jupyter environment with a GPU. Required libraries (PyTorch, torchvision, pywt, grad-cam, scipy, pandas) are installed via pip cells within the notebook itself.
4. Training is stochastic — exact numbers may vary slightly (±1-2%) between runs; the qualitative finding (accuracy and FFAS are largely independent) is stable across runs.

## Methodology summary

- **Preprocessing:** raw vibration signal → 1024-sample overlapping windows (50%) → Continuous Wavelet Transform (Morlet, scales 1–127) → 224×224 spectrogram image
- **Models:** ResNet50 and ViT-B16, ImageNet-pretrained, fine-tuned for 20 epochs (Adam, lr 1e-4)
- **Explainability:** Grad-CAM (last convolutional layer) for ResNet50; CLS-token-to-patch self-attention similarity (last layer) for ViT-B16

## Related work

This project extends the quantitative frequency-grounded explainability approach introduced by **Li et al. (2023), "Multilayer Grad-CAM," Journal of Manufacturing Systems**, which first proposed checking whether CNN activation maps concentrate at fault characteristic frequencies. PGXAI extends this with an energy-based alignment score, random-baseline validation with formal significance testing, and — to our knowledge — the first comparison of physics-grounded explainability between a CNN and a Vision Transformer.

## Limitations

- Validated on a single dataset (CWRU); components are architecture- and dataset-agnostic by design, but broader validation is future work.
- Only two saliency methods evaluated (Grad-CAM, ViT self-attention).
- FFAS tolerance (±10%) and harmonic count (2) fixed a priori from standard bearing-diagnostics convention, not empirically tuned.

## License

Academic project — MSc Computing (AI), Dublin City University, 2026.
