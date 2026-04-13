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

Stage 3 was designed as a compact controlled study of the corrected QANet implementation. All runs used the same mini-SQuAD preprocessing outputs, random seed `42`, batch size `8`, `3000` training steps, checkpoint interval `500`, and a monitoring protocol of `25` train batches and `25` dev batches per checkpoint. The baseline configuration used Adam, the lambda warmup scheduler, `qa_nll`, LayerNorm, dropout `0.1`, character dropout `0.05`, ReLU, and Kaiming initialization. Each ablation changed only one mechanism relative to that baseline.

These experiments should be interpreted as comparative evidence rather than final benchmark results. In particular, the reported Stage 3 dev scores come from a 25-batch sampled evaluation rather than the full dev set. This makes the relative ranking of configurations informative, but it also means the exact numbers are noisier than a full-dev evaluation.

### 5.1 Stage 3 Summary

Table 1 summarizes the compact Stage 3 sweep. All five runs completed successfully and none showed numerical instability or `nan` loss values.

| Experiment | Mechanism changed | Best dev F1 | Best dev EM | Final dev loss | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_adam_lambda_layernorm_dropout010` | Reference configuration | `13.6660` | `5.0` | `3.6976` | Stable baseline for comparison |
| `optimizer_sgdmomentum_step` | Optimizer + scheduler | `8.1885` | `2.5` | `4.5129` | Worse convergence than the baseline |
| `normalization_groupnorm` | LayerNorm -> GroupNorm | `14.9810` | `9.5` | `3.7029` | Modest improvement over baseline |
| `dropout_zero` | Remove dropout | `28.6822` | `21.5` | `3.0501` | Strongest sampled performance |
| `dropout_high` | Increase dropout | `5.0913` | `2.5` | `4.5232` | Clear underfitting / over-regularization |

The most striking pattern is that dropout strength had the largest effect under this short training budget. Removing dropout substantially improved sampled dev performance, while increasing dropout harmed both loss and F1/EM. GroupNorm produced a smaller but still consistent improvement over LayerNorm, while the SGD-momentum-plus-step variant underperformed Adam with warmup.

### 5.2 Experiment 1: Optimizer and Scheduler Choice

**Hypothesis.** Adam with the corrected lambda warmup scheduler should outperform SGD with momentum and a step scheduler, because Adam adapts learning rates per parameter and the warmup schedule avoids poor early-step behavior.

**Design.** The baseline configuration used `optimizer_name="adam"` and `scheduler_name="lambda"`. The controlled variant replaced these with `optimizer_name="sgd_momentum"` and `scheduler_name="step"` using `lr_step_size=1000` and `lr_gamma=0.5`. All other settings were unchanged.

**Results.**

| Optimizer | Scheduler | Best dev F1 | Best dev EM | Final dev loss | Final learning rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Adam | Lambda warmup | `13.6660` | `5.0` | `3.6976` | `8.6935e-4` |
| SGD momentum | Step | `8.1885` | `2.5` | `4.5129` | `1.2500e-4` |

**Analysis.** The baseline Adam plus warmup configuration outperformed the SGD-momentum-plus-step variant on every reported dev metric. Best dev F1 dropped from `13.6660` to `8.1885`, best dev EM fell from `5.0` to `2.5`, and final dev loss increased from `3.6976` to `4.5129`. This supports the hypothesis that the corrected Adam implementation and warmup schedule are better suited to the QANet training dynamics in this short-budget setting. A likely explanation is that Adam's adaptive updates make optimization easier in the presence of stacked attention and convolution blocks, while the step schedule decayed the SGD learning rate to a relatively small final value before the model had converged.

### 5.3 Experiment 2: Normalization Strategy

**Hypothesis.** LayerNorm was initially expected to perform best because QANet is an attention-heavy sequence model, and LayerNorm is a standard choice in that setting. However, GroupNorm might still help because the encoder also relies heavily on convolutional sublayers.

**Design.** The baseline used `norm_name="layer_norm"`. The controlled variant changed only the normalization type to `norm_name="group_norm"` with `norm_groups=8`, while keeping the optimizer, scheduler, dropout, and initialization unchanged.

**Results.**

| Normalization | Best dev F1 | Best dev EM | Final dev F1 | Final dev EM | Final dev loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| LayerNorm | `13.6660` | `5.0` | `11.1724` | `2.5` | `3.6976` |
| GroupNorm | `14.9810` | `9.5` | `13.1987` | `6.0` | `3.7029` |

**Analysis.** GroupNorm performed slightly but consistently better than LayerNorm in this sweep. Its best dev F1 improved from `13.6660` to `14.9810`, and best dev EM almost doubled from `5.0` to `9.5`. Final dev F1 and EM were also higher. The dev loss values were very close, so the main difference was in answer quality rather than a dramatic change in optimization stability. This result is interesting because it runs against the initial expectation that LayerNorm would be the natural fit for an attention-based architecture. A cautious interpretation is that, in this corrected implementation and with this small-batch regime, GroupNorm interacts favorably with the convolution-heavy encoder blocks. However, the margin is still modest enough that a full-dev rerun would be needed before making a strong claim.

### 5.4 Experiment 3: Regularization Strength Through Dropout

**Hypothesis.** Moderate dropout is normally expected to improve generalization, while excessive dropout should reduce both train and dev performance. Because the original dropout code was mathematically incorrect, this experiment was especially important after debugging.

**Design.** The baseline used `dropout=0.1` and `dropout_char=0.05`. Two controlled variants were tested: `dropout=0.0`, `dropout_char=0.0`, and `dropout=0.2`, `dropout_char=0.1`. Optimizer, scheduler, normalization, seed, and all data settings were fixed.

**Results.**

| Dropout | Char dropout | Best dev F1 | Best dev EM | Final train F1 | Final dev F1 | F1 gap (train - dev) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `0.0` | `28.6822` | `21.5` | `20.8615` | `28.6822` | `-7.8207` |
| `0.1` | `0.05` | `13.6660` | `5.0` | `7.7552` | `11.1724` | `-3.4171` |
| `0.2` | `0.1` | `5.0913` | `2.5` | `5.1382` | `4.1584` | `0.9798` |

**Analysis.** The dropout experiment produced the clearest result in the entire Stage 3 sweep. Removing dropout gave by far the best sampled dev performance, with best dev F1 increasing from `13.6660` to `28.6822` and best dev EM increasing from `5.0` to `21.5`. In contrast, increasing dropout to `0.2` / `0.1` caused a severe performance drop. This indicates that, at a training budget of only `3000` steps, the model is more limited by optimization than by overfitting. In other words, the baseline regularization appears too strong for such a short run, while the high-dropout setting clearly over-regularizes and prevents the model from fitting useful span patterns.

The negative train-dev F1 gaps in the `0.0` and baseline settings should not be over-interpreted as "better dev than train" in a true statistical sense. They are a consequence of using only 25 random train batches and 25 dev batches for monitoring, so the estimates are noisy. The main robust finding is the ranking of the three dropout settings: no dropout performed best, moderate dropout was clearly weaker, and high dropout was worst.

### 5.5 Overall Interpretation

Across the three mechanisms, the corrected QANet implementation was most sensitive to regularization strength, moderately sensitive to normalization choice, and also meaningfully affected by optimizer/scheduler selection. The best configuration tested in Stage 3 was `dropout=0.0`, `dropout_char=0.0` on top of the Adam-plus-lambda baseline. The normalization experiment suggests that GroupNorm may provide an additional improvement, although this was not combined with the no-dropout setting in the current sweep. If more compute were available, the most useful follow-up would be to evaluate a combined configuration using Adam, lambda warmup, GroupNorm, and zero dropout on the full dev set.

## 6. Discussion

The Stage 3 results reinforce the main lesson of the debugging phase: implementation correctness is necessary, but it is not sufficient for strong performance. Once the pipeline became stable, the remaining behavior depended strongly on optimization and regularization choices. This is exactly what should happen in a healthy experimental setup. Instead of fighting crashes, shape errors, and `nan` losses, the experiments revealed interpretable trade-offs between different deep learning mechanisms.

The strongest finding from the Stage 3 sweep is that the corrected model appears under-trained rather than over-regularized at `3000` steps. This explains why removing dropout helped substantially, while increasing dropout caused large drops in F1 and EM. It also helps explain why Adam with warmup beat SGD with a step schedule: under a tight training budget, faster and more adaptive optimization mattered more than conservative regularization.

The normalization result was more subtle. GroupNorm improved the sampled dev metrics over LayerNorm, but the improvement was not as large as the dropout effect. Therefore, GroupNorm is best described as a promising secondary enhancement rather than a definitive replacement. A full-dev evaluation and possibly a second seed would be needed before drawing a stronger conclusion.

One important limitation is that all Stage 3 runs used only a 25-batch dev monitor. These results are still useful for controlled comparison, but they are not final benchmark numbers. For the final report, the safest wording is that Stage 3 identifies likely beneficial design choices, not that it fully resolves model selection.

## 7. Conclusion

Stage 3 demonstrated that, after the major debugging fixes, the repository is not only executable but experimentally usable. The corrected QANet system can now support controlled ablations that isolate the effects of optimizers, schedulers, normalization, and dropout. This is important because it shows the codebase has been repaired to the point where meaningful empirical conclusions can be drawn.

Among the tested configurations, the baseline Adam-plus-lambda setup was stronger than the SGD-momentum-plus-step alternative, GroupNorm slightly outperformed LayerNorm, and dropout strength had the largest impact on performance. The strongest Stage 3 configuration was the no-dropout variant, which reached sampled best dev F1 `28.6822` and EM `21.5`, while the high-dropout condition performed worst. The most defensible overall conclusion is therefore that, in this short-budget setting, lighter regularization and adaptive optimization are more beneficial than stronger dropout or more conservative optimization schedules.

For final submission, the Stage 3 section can confidently argue that the repaired model supports controlled mechanism-level analysis. The only caveat is that the reported numbers come from sampled dev monitoring rather than full-dev evaluation. If time permits, the best next step is to rerun the strongest configuration on the full dev set so that the final report can pair its qualitative conclusions with a more stable quantitative result.

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
