# Multimodal Human-Centered AI for Sexism Identification (EXIST 2026)

## 1. Abstract
This system implements an **ICM-Optimized Multimodal Interaction Ensemble** for **EXIST 2026 Task 2.1**. Following the "Learning from Disagreement" (LeWiDi) paradigm, the model treats sexism as a probability distribution rather than a binary state. By utilizing **Kullback-Leibler (KL) Divergence** combined with **Supervised Contrastive Learning** and **Multi-Task Demographic Regularization**, the system directly optimizes for the **Information Contrastive Metric (ICM)**, capturing the nuanced disagreement of human annotators across memes.

---

## 2. System Architecture
The solution integrates textual, visual, demographic, and physiological processing through a deep cross-modal framework.

### **A. Core Encoders**
* **Textual:** `FacebookAI/xlm-roberta-base` augmented with Pre-computed OCR. Adapted via **LoRA** (TaskType: FEATURE_EXTRACTION) to prevent catastrophic forgetting.
* **Visual:** `openai/clip-vit-base-patch32` adapted via **LoRA** (Vision-specific projection targeting).
* **Demographic & Sensorial:** An `nn.Embedding` layer maps annotator metadata (Age, Gender, Study Level, Ethnicity), while a Tabular MLP ingests log-scaled physiological data (Eye-Tracking, EEG, Heart Rate).

### **B. Deep Cross-Modal Attention & Friction Fusion**
To model the complex relationship between text and imagery (e.g., sarcasm, irony), the system computes an 824-dimensional fused state $F$:
$$F = [\text{CrossAttn}(t, v), |t_{cls} - v_{cls}|, t_{cls} \odot v_{cls}, E_{demo}, E_{sensor}]$$
* **$\text{CrossAttn}(t, v)$:** Text sequences attending to visual patches via Multihead Attention to ground textual intent to visual regions.
* **$|t_{cls} - v_{cls}|$ & $t_{cls} \odot v_{cls}$:** Explicitly computes "Semantic Friction" and modality reinforcement.

---

## 3. Methodology

### **The Multi-Objective Loss Function**
The model optimizes a joint loss function to handle subjective disagreement while maintaining a sharp decision boundary:
$$L_{Total} = \alpha L_{KL\_Uncertainty} + \beta L_{Aux\_Gender} + \gamma L_{Contrastive}$$

1. **ICM-Aligned Objective ($L_{KL}$):** Minimizes the KL Divergence between predicted log-probabilities $Q$ and human distribution $P$. Modulated by an **Exponential Uncertainty Penalty** ($e^{-p(1-p)}$) to focus learning on high-consensus samples.
2. **Multi-Task Demographic Head ($L_{Aux}$):** An auxiliary Binary Cross-Entropy task that forces the network to predict the gender-specific "YES/NO" ratios, embedding cultural context directly into the shared backbone.
3. **Supervised Contrastive Learning ($L_{Contrastive}$):** Pulls memes with the same majority label together in the 824-dimensional space, pushing opposing memes apart to sharpen the boundary.

### **Stability Mechanisms**
* **LayerNorm Tuning:** Replaces BatchNorm in the main classifier for small-batch multimodal stability.
* **Modality Dropout:** Injects 15% path-level dropout to prevent lazy reliance on the dominant text modality.
* **Feature L2-Norm:** Contrastive embeddings are normalized before interaction to prevent gradient spikes.

---

## 4. Training and Evaluation

* **Threshold Calibration:** Instead of a fixed 0.50, we perform a dynamic grid search on the validation set at every epoch to find the **Optimal F1-Threshold** (Found consistently near 0.53).
* **Gradient Accumulation & Differential LR:** The LoRA encoders are trained with a fractional learning rate ($8e-6$) compared to the fusion heads ($3e-5$), with gradients accumulated to simulate a batch size of 64.
* **State-of-the-Art Ensemble Strategy:** A soft-voting blend combining the **Deep Multimodal Transformer (90%)** with a **Sensorial/Stylometric SVM (10%)**. The SVM utilizes demographic ratios, text stylometry (punctuation intensity, yelling ratio), and log-scaled physiological data to provide an algorithmic "sanity check" against deep learning hallucinations.

---

## 5. Summary of Upgrades
| Module                     | Baseline                     | ICM-Aligned Solution                                    |
|:---------------------------|:-----------------------------|:--------------------------------------------------------|
| **Backbone**               | Dual-Encoder (Concatenation) | XLM-R + CLIP + **Cross-Modal Attention**                |
| **Sensors & Demographics** | Ignored                      | Embedded directly via MLP & SVM                         |
| **Fusion**                 | Simple Concat                | CrossAttn + Semantic Friction ($u \odot v,  \| u-v \|$) |
| **Loss Function**          | BCEWithLogits                | **KL Divergence + Contrastive + Multi-Task**            |
| **Decision**               | Fixed 0.50                   | Calibrated Dynamic Thresholding                         |
| **Ensemble**               | Single Model                 | **90% Deep Net / 10% Feature SVM**                      |

---

## 6. Conclusion
By directly optimizing for the **ICM** through distribution matching, establishing deep cross-attention, and anchoring predictions with raw human biological and demographic data, this system provides a highly human-aligned approach to sexism identification. It effectively captures the physiological and information content of human disagreement, making it specifically tailored for the evaluation rigors of the EXIST 2026 competition.