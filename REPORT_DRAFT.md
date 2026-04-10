# COMP5329 / COMP4329 Assignment 1 Report Draft

**Title:** Debugging and Experimental Analysis of a QANet Question Answering System

**Group members:** [fill in names and student IDs]

**Code repository link:** [paste Google Drive public sharing link here]

## 1. Introduction

This assignment investigates a PyTorch implementation of QANet for extractive question answering on SQuAD v1.1. The task is to predict the start and end token positions of an answer span given a context paragraph and a natural language question. The provided codebase intentionally contained both fundamental pipeline errors and deeper mechanism-level implementation errors.

Our work had two goals. First, we repaired the training and evaluation pipeline so that the notebook could download/preprocess data, train the model, evaluate checkpoints, and reload saved weights without manual intervention. Second, we corrected the implementation of core deep learning mechanisms, including loss computation, optimization, learning-rate scheduling, normalization, dropout, initialization, convolution, and attention.

After debugging, the corrected training run showed meaningful learning behavior. In a 10,000-step run using Adam with the warmup lambda scheduler, the final training block reported loss `2.425453`; the monitored training subset achieved loss `1.864921`, F1 `38.140392`, and EM `28.750000`; and the monitored dev subset achieved loss `4.481346`, F1 `21.587100`, and EM `14.750000`, with best monitored dev F1 `21.5871` and best monitored EM `15.2500`. These results show that the model is no longer merely executable, but is learning non-trivial answer-span signals. The train-dev gap also suggests overfitting and limited generalization, which is expected given the small training setup and motivates controlled experiments on optimization and regularization.

**Important note for final submission:** the numbers above came from the 25-batch monitoring run. If you run full-dev evaluation with `test_num_batches=-1`, replace or supplement these values with the full-dev results.

## 2. Model and Experimental Setup

The model is QANet, consisting of word and character embeddings, convolutional embedding encoders, context-question attention, stacked model encoder blocks with self-attention and depthwise-separable convolution, and a pointer-style output head for start/end span prediction.

The preprocessing pipeline used:

- Dataset: SQuAD v1.1 mini training split from `_data/squad/train-mini.json` and dev set from `_data/squad/dev-v1.1.json`.
- Embeddings: mini GloVe file `_data/glove/glove.mini.txt`.
- Limits: context length `para_limit=400`, question length `ques_limit=50`, character limit `char_limit=16`.
- Core model settings: `d_model=96`, `num_heads=8`, `glove_dim=300`, `char_dim=64`, dropout `0.1`, character dropout `0.05`.

The main repaired training configuration used:

- Optimizer: `adam`.
- Scheduler: `lambda`, implementing warmup to the target learning rate.
- Loss: `qa_nll`.
- Number of training steps: `10,000`.
- Checkpoint interval: `500`.
- Monitoring evaluation: `25` train batches and `25` dev batches per checkpoint.
- Batch size: `[fill in exact batch size used in the final Colab run]`.
- Hardware: `[fill in T4 / L4 / other GPU used]`.

## 3. Debugging Analysis

The bugs fell into two categories. Stage I bugs prevented the system from running correctly end-to-end. Stage II bugs were deeper learning-mechanism errors that could allow execution but distort training dynamics, produce unstable values, or degrade generalization.

### 3.1 Stage I: Pipeline and Executability Fixes

The Stage I fixes restored basic functionality: argument passing, gradient computation, checkpoint compatibility, data-to-embedding routing, tensor shape conventions, and evaluation decoding. These changes allowed `assignment1.ipynb` to run a complete training/evaluation workflow.

| ID | Component | Bug impact | Correction |
| --- | --- | --- | --- |
| 1 | `TrainTools/train.py` | `argparse.Namespace` was constructed with a positional dictionary, so configuration fields were not accessible as attributes. | Unpacked locals with `argparse.Namespace(**{...})`. |
| 2 | `TrainTools/train_utils.py` | Backpropagation was called on `loss.item()`, a Python number with no computation graph. | Called `loss.backward()` and clipped gradients before `optimizer.step()`. |
| 3 | `Schedulers/scheduler.py` | The notebook requested `scheduler_name="none"`, but the registry did not include it. | Added a no-op scheduler entry. |
| 4 | `EvaluateTools/evaluate.py` | Evaluation loaded checkpoint key `"model"` while saving used `"model_state"`. | Loaded `ckpt["model_state"]`. |
| 5 | `Models/qanet.py` | Word IDs and character IDs were fed to the wrong embedding tables. | Sent `Cwid/Qwid` to `word_emb` and `Ccid/Qcid` to `char_emb`. |
| 6 | `Models/embedding.py` | Highway network transposed the batch dimension into the sequence axis. | Changed transpose from `(0, 2)` to `(1, 2)`. |
| 7 | `Models/embedding.py` | Character embeddings were permuted into an invalid `Conv2d` layout. | Permuted to `[B, d_char, L, char_len]`. |
| 8 | `Models/conv.py` | Custom `Conv1d` unfolded the channel dimension instead of sequence length. | Unfolded dimension `2`. |
| 9 | `Models/conv.py` | Custom `Conv2d` width padding used the pre-height-padding tensor height. | Built width padding using the updated height `x.size(2)`. |
| 10 | `Models/encoder.py` | Encoder block indexed normalization layers with `i + 1`, eventually going out of range. | Used the matching index `i`. |
| 11 | `Models/encoder.py` | Positional encoding frequency tensor had the wrong orientation. | Used a `[channels, 1]` layout. |
| 12 | `Models/encoder.py` | Multi-head attention reshaped/permuted tensors in the wrong order. | Used consistent `[batch, head, length, d_k]` handling before flattening heads. |
| 13 | `Models/heads.py` | Pointer head concatenated along the batch dimension. | Concatenated along the channel dimension. |
| 14 | `Models/attention.py` | Context-question attention multiplied attention and question matrices in the wrong order. | Computed `A = torch.bmm(S1, Q)`. |
| 15 | `Models/qanet.py` | Context and question masks were passed to context-question attention in the wrong order. | Passed `cmask` first and `qmask` second. |
| 16 | `EvaluateTools/eval_utils.py` | Evaluation used `argmax` across the batch dimension rather than sequence positions. | Used `argmax(..., dim=1)` for each example. |
| 17 | `Schedulers/scheduler.py` | The no-op scheduler used a local lambda that could not be pickled into checkpoints. | Replaced it with a top-level named identity function. |

### 3.2 Stage II: Deep Learning Mechanism Fixes

The Stage II fixes corrected the mathematical behavior of the learning system. These bugs were especially important because several of them did not necessarily crash the notebook, but they damaged the signal used for learning.

| ID | Component | Bug impact | Correction |
| --- | --- | --- | --- |
| 1 | `Losses/loss.py` | NLL loss passed labels and predictions in the wrong order for one span endpoint. | Used predictions first and labels second for both start and end losses. |
| 2 | `Schedulers/lambda_scheduler.py` | Lambda scheduler added its factor to the base LR instead of multiplying. | Returned `base_lr * factor`. |
| 3 | `Models/conv.py` | Depthwise-separable convolution ran pointwise before depthwise convolution. | Ran depthwise convolution first, then pointwise convolution. |
| 4 | `Models/encoder.py` | Encoder block computed attention but overwrote it with the residual. | Added the attention output to the residual. |
| 5 | `Models/Normalizations/layernorm.py` | LayerNorm used wrong broadcasting and swapped affine parameters. | Used `keepdim=True` and `x_norm * weight + bias`. |
| 6 | `Models/dropout.py` | Dropout divided by `p`, which amplified activations too strongly and caused instability. | Implemented inverted dropout using division by `1 - p`. |
| 7 | `Models/Activations/relu.py` | ReLU kept negative values and clipped positives incorrectly. | Implemented `max(0, x)`. |
| 8 | `Models/Activations/leakeyReLU.py` | LeakyReLU scaled positive values instead of negative values. | Scaled only the negative branch. |
| 9 | `Models/encoder.py` | Self-attention omitted the standard `1 / sqrt(d_k)` scaling. | Multiplied attention logits by the scale factor. |
| 10 | `Models/Normalizations/groupnorm.py` | GroupNorm reshaped channels and groups in the wrong order. | Reshaped to `[B, G, C/G, *spatial]`. |
| 11 | `Models/Initializations/kaiming.py` | Kaiming initialization used `sqrt(1 / fan)` instead of `sqrt(2 / fan)`. | Used the correct He initialization formula. |
| 12 | `Models/Initializations/xavier.py` | Xavier initialization used `fan_in * fan_out` instead of `fan_in + fan_out`. | Used the Glorot formula based on `fan_in + fan_out`. |
| 13 | `Optimizers/adam.py` | Adam stored moving-average state under different keys from the keys it read. | Used consistent `exp_avg` and `exp_avg_sq` keys. |
| 14 | `Optimizers/adam.py` | Adam bias correction used `beta * t` instead of `beta ** t`. | Used the exponential bias-correction terms. |
| 15 | `Optimizers/adam.py`, `Optimizers/sgd.py` | Weight decay used the wrong sign and pushed parameters in the wrong direction. | Added `wd * p` to gradients. |
| 16 | `Optimizers/sgd_momentum.py` | Momentum state was stored under one key and read under another. | Used a consistent `velocity` key. |
| 17 | `Optimizers/sgd_momentum.py` | Momentum velocity updated with the wrong sign. | Used `v = momentum * v + grad`. |
| 18 | `Schedulers/cosine_scheduler.py` | Cosine schedule missed the `0.5` factor and used the wrong constant name. | Implemented the standard cosine annealing equation. |
| 19 | `Schedulers/step_scheduler.py` | Step schedule used a linear expression rather than powers of `gamma`. | Used `base_lr * gamma ** floor(t / step_size)`. |
| 20 | `Models/heads.py`, `Losses/loss.py` | Output head returned log-probabilities while the loss registry also exposed cross-entropy expecting logits. | Made the head return raw masked logits; loss functions now apply the appropriate transformation. |
| 21 | `Optimizers/optimizer.py`, `Schedulers/scheduler.py` | Adam was configured with `lr=1.0` but the lambda scheduler was previously a no-op, producing unstable learning rates. | Implemented a warmup scheduler that outputs the intended effective LR. |
| 22 | `Optimizers/adam.py` | Adam second moment tracked raw gradients rather than squared gradients. | Used `addcmul_(grad, grad, ...)` to update the second moment. |
| 23 | `Models/encoder.py` | Self-attention masks were repeated across heads in the wrong batch order, allowing some flattened heads to use another example's padding mask. | Replaced `repeat(...)` with `repeat_interleave(self.num_heads, dim=0)` to match the flattened `[B * H, L, L]` attention order. |

These fixes explain why early runs either crashed or produced `nan` losses, while the corrected implementation achieved stable losses and non-zero F1/EM.

## 4. Evidence That Training Works

The corrected pipeline produced a full training run to step `10,000` and saved/evaluated checkpoints. The final logs showed:

| Step / phase | Loss | F1 | EM | Notes |
| --- | ---: | ---: | ---: | --- |
| Dev monitor before final checkpoint | `4.716447` | `19.150516` | `11.250000` | 25-batch dev monitor |
| Final training block, step `10,000` | `2.425453` | n/a | n/a | Training loss over final 500-step block |
| Final train monitor | `1.864921` | `38.140392` | `28.750000` | 25-batch train monitor |
| Final dev monitor | `4.481346` | `21.587100` | `14.750000` | 25-batch dev monitor |
| Best monitored dev result | n/a | `21.5871` | `15.2500` | Best values tracked during training |

This is sufficient evidence for the executability and trainability requirements. The training loop performed forward and backward propagation, updated parameters, evaluated on validation data, adjusted the learning rate, saved checkpoints, and completed without numerical instability. The model also improved from the earlier monitored dev F1 of `19.150516` to the final monitored F1 of `21.587100`.

However, these metrics should not be overstated. The dev loss remained much higher than the training monitor loss, and train F1/EM were higher than dev F1/EM. This indicates a generalization gap, likely due to the small training split, limited training budget, and the difficulty of span prediction. Therefore, the result supports the claim that the repaired model learns, but not the stronger claim that it is highly optimized.

## 5. Experimental Investigation

The assignment requires at least three controlled experiments. The following experiments are suitable because they isolate specific deep learning mechanisms implemented in the repository.

For fair comparison, all Stage 3 experiment runs should use the same data split, random seed, batch size, number of training steps, checkpoint interval, and monitoring protocol. The helper script `stage3_experiments.py` runs a compact controlled plan using `num_steps=3000`, `checkpoint=500`, `val_num_batches=25`, and `test_num_batches=25` by default. If you use that script, replace the placeholder table values below with the values from `_stage3_results/summary.csv`. Do not directly compare a 10,000-step baseline against 3,000-step ablations unless you explicitly label the comparison as unequal-budget.

### Experiment 1: Optimizer and Scheduler Choice

**Hypothesis.** Adam with warmup should converge faster and more stably than vanilla SGD because it adapts parameter-wise step sizes and avoids the unstable high effective learning rate that existed before the scheduler fix.

**Design.** Keep architecture, data split, batch size, seed, dropout, and evaluation protocol fixed. Compare:

- `optimizer_name="sgd"`, `scheduler_name="none"`.
- `optimizer_name="sgd_momentum"`, `scheduler_name="step"`.
- `optimizer_name="adam"`, `scheduler_name="lambda"`.

**Metrics.** Report train loss, dev loss, best dev F1, best dev EM, final learning rate, and whether `nan` or OOM occurred.

**Results.**

| Optimizer | Scheduler | Best dev F1 | Best dev EM | Final dev loss | Observation |
| --- | --- | ---: | ---: | ---: | --- |
| SGD | None | [fill in] | [fill in] | [fill in] | [fill in] |
| SGD momentum | Step | [fill in] | [fill in] | [fill in] | [fill in] |
| Adam | Lambda warmup | [fill in Stage 3 baseline] | [fill in] | [fill in] | Stable baseline run |

**Analysis draft.** Adam with lambda warmup is expected to be strongest because the corrected Adam implementation uses bias-corrected first and second moments, and the scheduler warms the effective learning rate to the target value. If the results show faster loss reduction or higher dev F1, this supports the hypothesis that adaptive optimization and warmup improve stability for the corrected QANet pipeline.

### Experiment 2: Normalization Strategy

**Hypothesis.** LayerNorm should perform better than GroupNorm in this sequence model because it normalizes over the feature/sequence representation of each example and is commonly used with attention-based architectures. GroupNorm may still stabilize convolutional blocks, but its grouping assumption may be less aligned with token-wise attention representations.

**Design.** Keep optimizer as Adam, scheduler as lambda, loss as `qa_nll`, and all architecture parameters fixed. Compare:

- `norm_name="layer_norm"`.
- `norm_name="group_norm"`, with `norm_groups=8`.

**Metrics.** Report best dev F1/EM, train-dev gap, and whether either normalization caused unstable loss.

**Results.**

| Normalization | Best dev F1 | Best dev EM | Train F1 | Train EM | Observation |
| --- | ---: | ---: | ---: | ---: | --- |
| LayerNorm | [fill in Stage 3 baseline] | [fill in] | [fill in] | [fill in] | Stable baseline run |
| GroupNorm | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |

**Analysis draft.** If LayerNorm achieves higher dev F1/EM, this suggests that normalizing each example over its full representation is better suited to the QANet encoder. If GroupNorm performs similarly, then the corrected implementation is robust to alternative normalization schemes.

### Experiment 3: Regularization Strength Through Dropout

**Hypothesis.** Moderate dropout should improve generalization compared with no dropout, but overly large dropout should reduce both train and dev performance by removing too much signal. This is especially relevant because the original dropout implementation was mathematically incorrect and caused unstable activation scaling.

**Design.** Keep optimizer, scheduler, normalization, seed, and evaluation protocol fixed. Compare:

- `dropout=0.0`, `dropout_char=0.0`.
- `dropout=0.1`, `dropout_char=0.05`.
- `dropout=0.2`, `dropout_char=0.1`.

**Metrics.** Report train loss, dev loss, train-dev F1 gap, best dev F1, and best dev EM.

**Results.**

| Dropout | Char dropout | Best dev F1 | Best dev EM | Train-dev gap | Observation |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0.0` | `0.0` | [fill in] | [fill in] | [fill in] | [fill in] |
| `0.1` | `0.05` | [fill in Stage 3 baseline] | [fill in] | [fill in] | Stable baseline run |
| `0.2` | `0.1` | [fill in] | [fill in] | [fill in] | [fill in] |

**Analysis draft.** The main run showed a train-dev F1 gap: train monitor F1 `38.140392` versus dev monitor F1 `21.587100`. If no dropout increases training F1 but lowers dev F1, it supports the role of dropout as a regularizer. If high dropout lowers both train and dev performance, it indicates underfitting.

## 6. Discussion

The debugging process showed that small implementation details can completely change the behavior of a deep learning system. Some bugs were explicit runtime errors, such as invalid tensor shapes, missing scheduler registry entries, incorrect checkpoint keys, and calling `.backward()` on `loss.item()`. Others were more subtle: incorrect Adam bias correction, incorrect second-moment updates, wrong dropout scaling, and inconsistent logits/log-probability assumptions could let the code run while producing unstable or misleading training.

The corrected implementation produced stable non-`nan` losses and non-zero F1/EM, indicating that the pipeline now satisfies the assignment's functional reliability requirement. The result is not yet an optimized SQuAD model, but it is a valid controlled platform for mechanism-oriented experiments.

The experimental plan focuses on optimizer/scheduler behavior, normalization strategy, and dropout regularization. These experiments are appropriate because they vary one mechanism at a time while controlling the data, architecture, seed, and evaluation protocol. This allows the report to discuss causal effects rather than presenting unstructured hyperparameter tuning.

## 7. Conclusion

We repaired the QANet framework at both the pipeline and deep learning mechanism levels. The corrected code can preprocess data, train the model, evaluate validation performance, and save/load checkpoints. The most important fixes included restoring correct gradient computation, tensor shape handling, attention masking, loss/logit consistency, dropout scaling, normalization behavior, initialization formulas, optimizer updates, and learning-rate schedules.

The final training evidence demonstrates that the model learns meaningful answer-span information, with monitored dev F1 reaching `21.5871` and monitored dev EM reaching `15.2500` in the 10,000-step run. The remaining train-dev gap suggests that future improvement should focus on training data size, regularization, and optimization settings rather than basic executability.

## Appendix A: Suggested Final Checklist

- Replace group member names and repository link on the first page.
- Run full-dev evaluation with `test_num_batches=-1` if time permits.
- Save final logs or screenshots as evidence.
- Fill in all experiment tables before submitting.
- Make sure the notebook uses the same training configuration reported here.
- Do not claim full-dev performance if the values came from `test_num_batches=25`.
- Export the final report using the official Word or LaTeX template.

## Appendix B: Stage 3 Runner

To generate the controlled Stage 3 results, run this in Colab after download and preprocessing. The default runner performs five runs: one baseline, one optimizer/scheduler variant, one normalization variant, and two dropout variants.

```python
from stage3_experiments import run_stage3_experiments

summary = run_stage3_experiments(
    common_overrides={
        "batch_size": 8,      # use 16 on L4 if memory allows
        "num_steps": 3000,
    },
    final_eval_batches=None,  # use -1 only if you want full-dev eval after every run
)
summary
```

This produces:

- `_stage3_results/summary.csv`: compact table for the report.
- `_stage3_results/<experiment_name>.json`: full history/config for each run.
- `_stage3_results/models/<experiment_name>/model.pt`: checkpoint for each run.

If time is limited, keep `final_eval_batches=None` during the ablation sweep, then run full-dev evaluation only for the best checkpoint.
