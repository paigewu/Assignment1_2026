# Debug Log

This document records the debugging work done on the assignment codebase and splits the fixes into the two stages described in the assignment brief.

- `Stage 1`: fixes needed to make the notebook and full pipeline run end-to-end
- `Stage 2`: fixes where the code might run, but the deep learning mechanism was implemented incorrectly

For each item, I list:

- `Where`: the file and function
- `From`: the original code or behavior
- `To`: the corrected code or behavior
- `Why`: a plain-English explanation

## Stage 1: Pipeline / Executability Fixes

These are the fixes most directly tied to “can `assignment1.ipynb` run training and evaluation without crashing?”

### 1. Training config namespace was built incorrectly

`Where`: [TrainTools/train.py](/Users/siyiwu/Desktop/Assignment1_2026/TrainTools/train.py#L107)

`From`

```python
args = argparse.Namespace({k: v for k, v in locals().items()})
```

`To`

```python
args = argparse.Namespace(**{k: v for k, v in locals().items()})
```

`Why`

The training code creates an `args` object so the rest of the pipeline can read configuration values like file paths and model sizes. The original line passed a dictionary as one positional argument instead of unpacking it into named fields. That causes the very first training call to fail.

In simple terms: the settings object was built in the wrong shape.

### 2. Backpropagation was called on `loss.item()` instead of the loss tensor

`Where`: [TrainTools/train_utils.py](/Users/siyiwu/Desktop/Assignment1_2026/TrainTools/train_utils.py#L34-L36)

`From`

```python
loss.item().backward()
optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

`To`

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()
```

`Why`

`loss.item()` converts the tensor loss into a plain Python number, which cannot be used for gradient computation. Also, clipping gradients should happen before the optimizer updates parameters.

In simple terms: the model was throwing away the information it needed in order to learn.

### 3. The notebook asked for a scheduler named `"none"`, but the code did not support it

`Where`: [Schedulers/scheduler.py](/Users/siyiwu/Desktop/Assignment1_2026/Schedulers/scheduler.py)

`From`

```python
schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
}
```

`To`

```python
def none_scheduler(optimizer, args):
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
    "none":    none_scheduler,
}
```

`Why`

The training cell in the notebook uses `scheduler_name="none"`. Without a registry entry for `"none"`, training crashes immediately before real learning even starts.

In simple terms: the notebook asked for “no scheduler,” but the code did not understand that option.

### 4. Evaluation loaded the wrong checkpoint key

`Where`: [EvaluateTools/evaluate.py](/Users/siyiwu/Desktop/Assignment1_2026/EvaluateTools/evaluate.py#L117-L119)

`From`

```python
model.load_state_dict(ckpt["model"])
```

`To`

```python
model.load_state_dict(ckpt["model_state"])
```

`Why`

The checkpoint saver stores weights under `"model_state"`, but evaluation tried to load `"model"`. Even a successful training run would fail at evaluation time.

In simple terms: the model was saved under one name and loaded under a different name.

### 5. Word IDs and character IDs were fed into the wrong embedding tables

`Where`: [Models/qanet.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/qanet.py#L65-L66)

`From`

```python
Cw, Cc = self.char_emb(Cwid), self.word_emb(Ccid)
Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
```

`To`

```python
Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
```

`Why`

`Cwid` and `Qwid` are word indices, while `Ccid` and `Qcid` are character indices. The original code mixed them up, so the model started with the wrong lookup representations.

In simple terms: it tried to read words from the character dictionary and characters from the word dictionary.

### 6. The highway network used the wrong transpose

`Where`: [Models/embedding.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/embedding.py#L19)

`From`

```python
x = x.transpose(0, 2)
```

`To`

```python
x = x.transpose(1, 2)
```

`Why`

The tensor should move from `[batch, channels, length]` to `[batch, length, channels]`. Swapping dimensions `0` and `2` moves the batch dimension into the wrong place and breaks the data layout.

### 7. Character embeddings were permuted into the wrong Conv2D layout

`Where`: [Models/embedding.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/embedding.py#L37)

`From`

```python
ch_emb = ch_emb.permute(0, 2, 1, 3)
```

`To`

```python
ch_emb = ch_emb.permute(0, 3, 1, 2)
```

`Why`

The character tensor starts as `[batch, seq_len, char_len, char_dim]`, but `Conv2d` expects `[batch, channels, height, width]`. The channel dimension should be `char_dim`, not `char_len`.

In simple terms: the character tensor was being fed into the convolution sideways.

### 8. Custom 1D convolution unfolded the wrong dimension

`Where`: [Models/conv.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/conv.py#L55)

`From`

```python
x_unf = x.unfold(1, self.kernel_size, 1)
```

`To`

```python
x_unf = x.unfold(2, self.kernel_size, 1)
```

`Why`

For a tensor shaped `[batch, channels, length]`, convolution should slide across the length dimension, which is dimension `2`, not the channel dimension.

### 9. Custom 2D convolution built width padding with the wrong height

`Where`: [Models/conv.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/conv.py#L124)

`From`

```python
pad_w = x.new_zeros(B, C_in, H, p)
```

`To`

```python
pad_w = x.new_zeros(B, C_in, x.size(2), p)
```

`Why`

After adding height padding, the tensor height is no longer `H`. Using the old height causes the side padding tensor to have the wrong shape and can break concatenation.

### 10. Encoder block indexed normalization layers incorrectly

`Where`: [Models/encoder.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/encoder.py#L121)

`From`

```python
out = self.norms[i + 1](out)
```

`To`

```python
out = self.norms[i](out)
```

`Why`

If there are `conv_num` convolution layers, there are `conv_num` matching normalization layers indexed from `0` to `conv_num - 1`. Using `i + 1` eventually tries to access past the end of the list.

### 11. Positional encoding shape setup was wrong

`Where`: [Models/encoder.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/encoder.py#L29-L38)

`From`

```python
freqs = torch.tensor(...).unsqueeze(0)
```

`To`

```python
freqs = torch.tensor(...).unsqueeze(1)
```

`Why`

The positional encoding construction expects a `[channels, 1]` layout. The original layout was sideways and broke the positional encoding math.

### 12. Attention tensors were reshaped in the wrong order

`Where`: [Models/encoder.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/encoder.py#L70-L85)

`From`

```python
q = q.permute(2, 0, 1, 3)
...
out = out.permute(1, 2, 0, 3)
```

`To`

```python
q = q.permute(0, 2, 1, 3)
...
out = out.permute(0, 2, 1, 3)
```

`Why`

Multi-head attention needs a consistent ordering of batch, head, sequence length, and head dimension. The original code mixed those dimensions and corrupted the attention calculation.

### 13. Pointer head concatenated along the wrong dimension

`Where`: [Models/heads.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/heads.py#L23)

`From`

```python
X1 = torch.cat([M1, M2], dim=0)
```

`To`

```python
X1 = torch.cat([M1, M2], dim=1)
```

`Why`

The pointer layer should combine feature channels for the same example, not stack different examples into a larger batch.

### 14. Context-question attention multiplied matrices in the wrong order

`Where`: [Models/attention.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/attention.py#L38)

`From`

```python
A = torch.bmm(Q, S1)
```

`To`

```python
A = torch.bmm(S1, Q)
```

`Why`

The attention map `S1` represents how context positions attend to question positions, so it should left-multiply the question representation. The old order was backwards.

### 15. Context and question masks were passed into attention in the wrong order

`Where`: [Models/qanet.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/qanet.py#L75)

`From`

```python
X = self.cq_att(Ce, Qe, qmask, cmask)
```

`To`

```python
X = self.cq_att(Ce, Qe, cmask, qmask)
```

`Why`

The attention module expects the context mask first and the question mask second. Swapping them means padding is masked incorrectly.

### 16. Evaluation chose answer positions along the wrong dimension

`Where`: [EvaluateTools/eval_utils.py](/Users/siyiwu/Desktop/Assignment1_2026/EvaluateTools/eval_utils.py#L107-L108)

`From`

```python
yp1 = torch.argmax(p1, dim=0)
yp2 = torch.argmax(p2, dim=0)
```

`To`

```python
yp1 = torch.argmax(p1, dim=1)
yp2 = torch.argmax(p2, dim=1)
```

`Why`

The model output shape is `[batch_size, sequence_length]`. We want the best position inside each example, so the argmax must be taken across the sequence dimension, not across the batch dimension.

### 17. The no-op scheduler used a lambda that could not be pickled into checkpoints

`Where`: [Schedulers/scheduler.py](/Users/siyiwu/Desktop/Assignment1_2026/Schedulers/scheduler.py)

`From`

```python
return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
```

`To`

```python
def _identity_lr_lambda(_):
    return 1.0

return LambdaLR(optimizer, lr_lambda=_identity_lr_lambda)
```

`Why`

Training saves the optimizer and scheduler state into checkpoints. Python cannot pickle a locally defined `lambda`, so checkpoint saving crashed with:

```python
AttributeError: Can't pickle local object ...
```

Using a named top-level function fixes that.

In simple terms: the scheduler worked during training, but it could not be saved because anonymous functions are not safely serializable here.

### 18. Training always overwrote the checkpoint, even when dev performance got worse

`Where`: [TrainTools/train.py](/Users/siyiwu/Desktop/Assignment1_2026/TrainTools/train.py)

`From`

```python
best_f1  = 0.0
best_em  = 0.0
...
if dev_f1 < best_f1 and dev_em < best_em:
    ...
else:
    patience = 0
    best_f1  = max(best_f1, dev_f1)
    best_em  = max(best_em, dev_em)

save_checkpoint(...)
```

`To`

```python
best_f1  = -1.0
best_em  = -1.0
...
improved = (dev_f1 > best_f1) or (dev_f1 == best_f1 and dev_em > best_em)

if improved:
    patience = 0
    best_f1 = dev_f1
    best_em = dev_em
    save_checkpoint(...)
elif dev_f1 < best_f1 and dev_em < best_em:
    ...
else:
    patience = 0
```

`Why`

The original loop saved a checkpoint at every evaluation block, even if the current model was worse than an earlier one. That means the final file on disk could easily be a weaker checkpoint than the best model seen during training.

In simple terms: training kept overwriting the good model with later worse ones.

## Stage 2: Deep Learning Mechanism Fixes

These changes are less about “can the notebook run?” and more about “are the deep learning components implemented correctly?”

### 1. NLL loss used its inputs in the wrong order

`Where`: [Losses/loss.py](/Users/siyiwu/Desktop/Assignment1_2026/Losses/loss.py#L7)

`From`

```python
return 0.5 * (F.nll_loss(y1, p1) + F.nll_loss(p2, y2))
```

`To`

```python
return 0.5 * (F.nll_loss(p1, y1) + F.nll_loss(p2, y2))
```

`Why`

PyTorch expects predictions first and labels second. The original start-position loss reversed that order.

### 2. Lambda scheduler added instead of multiplied

`Where`: [Schedulers/lambda_scheduler.py](/Users/siyiwu/Desktop/Assignment1_2026/Schedulers/lambda_scheduler.py#L20-L23)

`From`

```python
return [base_lr + factor for base_lr in self.base_lrs]
```

`To`

```python
return [base_lr * factor for base_lr in self.base_lrs]
```

`Why`

A lambda scheduler is supposed to scale the base learning rate by a factor, not add the factor to the learning rate.

### 3. Depthwise-separable convolution ran in the wrong order

`Where`: [Models/conv.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/conv.py#L174-L175)

`From`

```python
return self.depthwise_conv(self.pointwise_conv(x))
```

`To`

```python
return self.pointwise_conv(self.depthwise_conv(x))
```

`Why`

Depthwise-separable convolution should first do the depthwise operation, then the pointwise mixing. Reversing the order changes the intended behavior.

### 4. Encoder block threw away the attention output

`Where`: [Models/encoder.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/encoder.py#L123-L125)

`From`

```python
out = self.self_att(out, mask)
out = res
out = self.drop(out)
```

`To`

```python
out = self.self_att(out, mask)
out = out + res
out = self.drop(out)
```

`Why`

The old code computed attention and then immediately overwrote the result with the residual tensor, effectively discarding the attention block.

### 5. LayerNorm used the wrong broadcasting behavior and affine formula

`Where`: [Models/Normalizations/layernorm.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/Normalizations/layernorm.py#L34-L40)

`From`

```python
mean = x.mean(dim=dims, keepdim=False)
var = x.var(dim=dims, keepdim=False, unbiased=False)
...
return x_norm * self.bias + self.weight
```

`To`

```python
mean = x.mean(dim=dims, keepdim=True)
var = x.var(dim=dims, keepdim=True, unbiased=False)
...
return x_norm * self.weight + self.bias
```

`Why`

The normalization statistics should keep their dimensions for clean broadcasting, and the affine transformation must be `x_norm * weight + bias`, not the reverse.

### 6. Dropout used the wrong scaling factor and could cause exploding activations

`Where`: [Models/dropout.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/dropout.py#L13-L17)

`From`

```python
return x * mask / self.p
```

`To`

```python
return x * mask / (1.0 - self.p)
```

`Why`

In inverted dropout, surviving activations should be scaled by `1 / (1 - p)`. Dividing by `p` instead can magnify activations massively and is a strong reason for `nan` losses.

### 7. ReLU was implemented backwards

`Where`: [Models/Activations/relu.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/Activations/relu.py#L11-L12)

`From`

```python
return x.clamp(max=0.0)
```

`To`

```python
return x.clamp(min=0.0)
```

`Why`

ReLU should keep positive values and zero out negative ones. The original code did the opposite.

### 8. LeakyReLU was implemented backwards

`Where`: [Models/Activations/leakeyReLU.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/Activations/leakeyReLU.py#L18-L19)

`From`

```python
return torch.where(x < 0, x, self.negative_slope * x)
```

`To`

```python
return torch.where(x < 0, self.negative_slope * x, x)
```

`Why`

LeakyReLU should scale negative values slightly and keep positive values unchanged. The original code reversed those two branches.

### 9. Self-attention was missing the standard scaling factor

`Where`: [Models/encoder.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/encoder.py#L78)

`From`

```python
attn = torch.bmm(q, k.transpose(1, 2))
```

`To`

```python
attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
```

`Why`

Scaled dot-product attention divides by `sqrt(d_k)`. Without this, attention scores can become too large and destabilize softmax.

### 10. GroupNorm reshaped channels into groups in the wrong order

`Where`: [Models/Normalizations/groupnorm.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/Normalizations/groupnorm.py#L31)

`From`

```python
x = x.view(B, C // self.G, self.G, *spatial)
```

`To`

```python
x = x.view(B, self.G, C // self.G, *spatial)
```

`Why`

Group normalization should split channels into `num_groups` and `channels_per_group`. The original reshape swapped those two ideas.

### 11. Kaiming initialization used the wrong formula

`Where`: [Models/Initializations/kaiming.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/Initializations/kaiming.py)

`From`

```python
std = math.sqrt(1.0 / fan)
```

`To`

```python
std = math.sqrt(2.0 / fan)
```

`Why`

Kaiming initialization for ReLU-style networks uses `sqrt(2 / fan)`, not `sqrt(1 / fan)`.

### 12. Xavier initialization used multiplication instead of addition

`Where`: [Models/Initializations/xavier.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/Initializations/xavier.py)

`From`

```python
std = gain * math.sqrt(2.0 / (fan_in * fan_out))
```

`To`

```python
std = gain * math.sqrt(2.0 / (fan_in + fan_out))
```

`Why`

Xavier initialization depends on `fan_in + fan_out`, not their product.

### 13. Adam used the wrong state keys

`Where`: [Optimizers/adam.py](/Users/siyiwu/Desktop/Assignment1_2026/Optimizers/adam.py#L58-L63)

`From`

```python
state["exp_avg"] = torch.zeros_like(p)
state["exp_avg_sq"] = torch.zeros_like(p)
...
m, v = state["m"], state["v"]
```

`To`

```python
state["exp_avg"] = torch.zeros_like(p)
state["exp_avg_sq"] = torch.zeros_like(p)
...
m, v = state["exp_avg"], state["exp_avg_sq"]
```

`Why`

The optimizer stored its moving averages under one pair of names and then tried to read them back using different names.

### 14. Adam used the wrong bias-correction formula

`Where`: [Optimizers/adam.py](/Users/siyiwu/Desktop/Assignment1_2026/Optimizers/adam.py#L71-L75)

`From`

```python
bias_correction1 = 1.0 - beta1 * t
bias_correction2 = 1.0 - beta2 * t
```

`To`

```python
bias_correction1 = 1.0 - beta1 ** t
bias_correction2 = 1.0 - beta2 ** t
```

`Why`

Adam’s bias correction is based on powers of `beta`, not simple multiplication by the step index.

### 15. Weight decay sign was wrong in Adam and SGD

`Where`: [Optimizers/adam.py](/Users/siyiwu/Desktop/Assignment1_2026/Optimizers/adam.py#L51-L53), [Optimizers/sgd.py](/Users/siyiwu/Desktop/Assignment1_2026/Optimizers/sgd.py#L33-L36)

`From`

```python
grad = grad.add(p, alpha=-wd)
```

`To`

```python
grad = grad.add(p, alpha=wd)
```

`Why`

Standard L2-style weight decay adds `wd * p` to the gradient. The original negative sign pushed in the wrong direction.

### 16. SGD with momentum stored velocity under one name and read it under another

`Where`: [Optimizers/sgd_momentum.py](/Users/siyiwu/Desktop/Assignment1_2026/Optimizers/sgd_momentum.py#L47-L51)

`From`

```python
if "velocity" not in state:
    state["vel"] = torch.zeros_like(p)

v = state["velocity"]
```

`To`

```python
if "velocity" not in state:
    state["velocity"] = torch.zeros_like(p)

v = state["velocity"]
```

`Why`

This is the same kind of bookkeeping bug as in Adam: the optimizer wrote to one key and read from another.

### 17. SGD with momentum updated velocity with the wrong sign

`Where`: [Optimizers/sgd_momentum.py](/Users/siyiwu/Desktop/Assignment1_2026/Optimizers/sgd_momentum.py#L53-L56)

`From`

```python
v.mul_(mu).sub_(grad)
p.add_(v, alpha=-lr)
```

`To`

```python
v.mul_(mu).add_(grad)
p.add_(v, alpha=-lr)
```

`Why`

The intended update rule in the file comments is `v = momentum * v + grad`. The original code subtracted the gradient instead.

### 18. Cosine scheduler formula was wrong

`Where`: [Schedulers/cosine_scheduler.py](/Users/siyiwu/Desktop/Assignment1_2026/Schedulers/cosine_scheduler.py#L27-L29)

`From`

```python
self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.PI * t / self.T_max))
```

`To`

```python
self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max))
```

`Why`

The standard cosine annealing formula includes the factor `0.5`, and Python uses `math.pi`, not `math.PI`.

### 19. Step scheduler formula was wrong

`Where`: [Schedulers/step_scheduler.py](/Users/siyiwu/Desktop/Assignment1_2026/Schedulers/step_scheduler.py#L24-L26)

`From`

```python
base_lr * self.gamma * (t // self.step_size)
```

`To`

```python
base_lr * (self.gamma ** (t // self.step_size))
```

`Why`

Step decay should use powers of `gamma`, not linear multiplication by the step count.

### 20. The pointer head and loss functions were internally inconsistent

`Where`: [Models/heads.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/heads.py#L24-L28), [Losses/loss.py](/Users/siyiwu/Desktop/Assignment1_2026/Losses/loss.py#L4-L16)

`From`

```python
Y1 = mask_logits(Y1, mask)
Y2 = mask_logits(Y2, mask)
p1 = F.log_softmax(Y1, dim=1)
p2 = F.log_softmax(Y2, dim=1)
return p1, p2
```

and

```python
def qa_ce_loss(p1, p2, y1, y2):
    return F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)
```

`To`

```python
Y1 = mask_logits(Y1, mask)
Y2 = mask_logits(Y2, mask)
return Y1, Y2
```

and

```python
def qa_nll_loss(p1, p2, y1, y2):
    return 0.5 * (
        F.nll_loss(F.log_softmax(p1, dim=1), y1)
        + F.nll_loss(F.log_softmax(p2, dim=1), y2)
    )
```

`Why`

The model head always returned log-probabilities, but the repository also exposed a cross-entropy loss option that expects raw logits. That made the loss registry inconsistent: `qa_nll` and `qa_ce` were not operating on the same type of model output.

The corrected design is cleaner:

- the model head returns raw masked logits
- `qa_nll_loss` applies `log_softmax` itself
- `qa_ce_loss` uses the same raw logits directly

In simple terms: the model and the loss functions were speaking slightly different “languages” about what the outputs meant.

### 21. The Adam optimizer and lambda scheduler contract was broken

`Where`: [Optimizers/optimizer.py](/Users/siyiwu/Desktop/Assignment1_2026/Optimizers/optimizer.py#L8-L18), [Schedulers/scheduler.py](/Users/siyiwu/Desktop/Assignment1_2026/Schedulers/scheduler.py#L38-L47)

`From`

The optimizer comments say:

```python
adam sets lr=1.0 because its learning rate is entirely controlled by
the paired warmup_lambda scheduler
```

but the scheduler implementation was effectively:

```python
return LambdaLR(optimizer, lr_lambda=_identity_lr_lambda)
```

`To`

```python
return LambdaLR(
    optimizer,
    lr_lambda=partial(
        _warmup_lr_lambda,
        learning_rate=getattr(args, "learning_rate", 1e-3),
        warmup_steps=getattr(args, "lr_step_size", 1000),
    ),
)
```

`Why`

There was a mismatch between the optimizer design and the scheduler implementation:

- `adam` was configured with base learning rate `1.0`
- the comments clearly expected a warmup scheduler to output the real learning rate
- but the actual `"lambda"` scheduler was just a constant no-op

That means the Adam path would effectively run at learning rate `1.0`, which is extremely likely to destabilize training.

In simple terms: the optimizer and scheduler were meant to work as a pair, but one half of that pair was missing.

### 22. Adam second moment tracked gradient instead of squared gradient

`Where`: [Optimizers/adam.py](Optimizers/adam.py#L69)

`From`

```python
v.mul_(beta2).add_(grad, alpha=1.0 - beta2)
```

`To`

```python
v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
```

`Why`

Adam's second moment `v` is supposed to track the exponential moving average of the **squared** gradient (`grad²`), which approximates the variance. Using plain `grad` instead means `v` tracks the mean gradient rather than the variance, so the adaptive scaling that makes Adam effective is completely wrong. This causes the optimizer to take poorly-scaled steps for every parameter throughout training.


When the notebook printed `STEP 10 loss nan`, that was not a sign of healthy learning. It meant the code had progressed further into training, but some deep learning mechanism was still mathematically wrong.

The strongest `nan` suspect in this repository was the dropout bug, because it could enlarge activations by a factor of `10` when `p = 0.1`. Combined with a deep architecture and attention layers, that is a plausible reason for unstable values.

### 23. Preprocessing used an arbitrary answer span when multiple gold spans existed

`Where`: [Tools/preproc.py](/Users/siyiwu/Desktop/Assignment1_2026/Tools/preproc.py)

`From`

```python
or (ex["y2s"][0] - ex["y1s"][0]) > ans_limit
...
y1s.append(example["y1s"][-1])
y2s.append(example["y2s"][-1])
```

`To`

```python
def choose_answer_span(ex):
    spans = list(zip(ex["y1s"], ex["y2s"]))
    if not spans:
        return None
    return min(spans, key=lambda t: ((t[1] - t[0] + 1), t[0]))
...
or (y2 - y1 + 1) > ans_limit
...
y1s.append(y1)
y2s.append(y2)
```

`Why`

Some SQuAD examples contain multiple gold answer spans. The original code filtered examples using one span but then trained on a different span, and that chosen target was simply the last annotation in the list. That makes supervision inconsistent and noisier than it needs to be.

The fix chooses one canonical target span per example, preferring the shortest valid span and then the earliest one.

In simple terms: the model was sometimes being trained against a moving answer target for the same question.

### 24. Weight decay was applied too broadly across all parameter types

`Where`: [TrainTools/train.py](/Users/siyiwu/Desktop/Assignment1_2026/TrainTools/train.py)

`From`

```python
params    = (p for p in model.parameters() if p.requires_grad)
optimizer = optimizers[optimizer_name](params, args)
```

`To`

```python
decay_params = []
no_decay_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
        no_decay_params.append(p)
    else:
        decay_params.append(p)

param_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
]
optimizer = optimizers[optimizer_name](param_groups, args)
```

`Why`

Applying L2-style weight decay to every parameter is usually a poor fit for bias terms and normalization parameters. Those parameters control offsets and scaling rather than main weight matrices, so regularizing them the same way can hurt optimization.

This change keeps weight decay on the main learned weight tensors while excluding bias and normalization parameters.

In simple terms: the regularizer was penalizing some parameter types that usually should be left alone.

### 25. Evaluation decoded start and end independently instead of choosing the best valid span

`Where`: [EvaluateTools/eval_utils.py](/Users/siyiwu/Desktop/Assignment1_2026/EvaluateTools/eval_utils.py)

`From`

```python
yp1 = torch.argmax(p1, dim=1)
yp2 = torch.argmax(p2, dim=1)
yps = torch.stack([yp1, yp2], dim=1)
ymin, _ = torch.min(yps, dim=1)
ymax, _ = torch.max(yps, dim=1)
```

`To`

```python
scores = p1.unsqueeze(2) + p2.unsqueeze(1)
valid = torch.triu(torch.ones_like(scores, dtype=torch.bool))
scores = scores.masked_fill(~valid, float("-inf"))
flat_idx = scores.view(scores.size(0), -1).argmax(dim=1)
seq_len = p1.size(1)
yp1 = flat_idx // seq_len
yp2 = flat_idx % seq_len
```

`Why`

The original evaluation picked the best start index and best end index independently, then just reordered them if the end came before the start. That is not how span-based QA decoding is usually defined. QANet-style decoding should choose the single best valid `(start, end)` pair with the constraint `start <= end`.

This change does not alter the trained model itself. It corrects how the model's start/end scores are turned into one final answer span during evaluation.

In simple terms: evaluation should pick the best whole answer span, not the best start and end separately.

## Architecture / Design Changes For Better Performance

These changes are intentionally separated from Stage 1 and Stage 2. They are not being claimed as obvious bugs in the original template. Instead, they are architecture/design choices that move the implementation closer to the original QANet paper and may improve generalization.

### A. Pretrained word embeddings are now frozen by default

`Where`: [Models/qanet.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/qanet.py)

`From`

```python
self.word_emb = nn.Embedding.from_pretrained(
    torch.tensor(word_mat, dtype=torch.float32),
    freeze=False
)
```

`To`

```python
freeze_word = bool(getattr(args, "freeze_word", True))
...
self.word_emb = nn.Embedding.from_pretrained(
    torch.tensor(word_mat, dtype=torch.float32),
    freeze=freeze_word
)
```

`Why`

The original QANet setup commonly keeps pretrained GloVe embeddings fixed. Freezing them preserves the pretrained semantic space and can reduce overfitting, especially when the model is trained for relatively few steps.

In simple terms: the model now treats pretrained word vectors more like a stable knowledge source than another set of weights to memorize with.

### B. Context and question now share the same projection convolution

`Where`: [Models/qanet.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/qanet.py)

`From`

```python
self.context_conv = DepthwiseSeparableConv(...)
self.question_conv = DepthwiseSeparableConv(...)
...
C = self.context_conv(C)
Q = self.question_conv(Q)
```

`To`

```python
self.input_proj = DepthwiseSeparableConv(...)
...
C = self.input_proj(C)
Q = self.input_proj(Q)
```

`Why`

Sharing the same projection layer pushes context and question representations into a more aligned feature space before cross-attention. This is closer to the paper-style idea that both sequences should be encoded in a comparable way.

In simple terms: context words and question words are now translated into the same “internal language” before they are compared.

### C. Context and question now share the same embedding encoder block

`Where`: [Models/qanet.py](/Users/siyiwu/Desktop/Assignment1_2026/Models/qanet.py)

`From`

```python
self.c_emb_enc = EncoderBlock(...)
self.q_emb_enc = EncoderBlock(...)
...
Ce = self.c_emb_enc(C, cmask)
Qe = self.q_emb_enc(Q, qmask)
```

`To`

```python
self.emb_enc = EncoderBlock(...)
...
Ce = self.emb_enc(C, cmask)
Qe = self.emb_enc(Q, qmask)
```

`Why`

Using one shared embedding encoder encourages context and question to be processed with the same inductive bias and reduces extra parameters that can overfit. This again moves the model closer to the original QANet design philosophy.

In simple terms: the model now uses the same reader for the paragraph and the question instead of teaching two separate readers from scratch.
