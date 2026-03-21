# ComfyUI-Spectrum-Qwen

A ComfyUI custom node that applies a Spectrum-style spectral hidden-state forecaster to **native Qwen Image-family models**, with the primary target being **Qwen-Image-Edit-2511**.

This repo patches the Qwen transformer core used by the loaded `MODEL`, captures the final **pre-`norm_out` image-stream hidden state** on real steps, fits a ridge-regularized Chebyshev model over recent real steps, and uses that forecast to skip selected expensive Qwen transformer passes.

## What this supports

Targeted support:

- ComfyUI native **Qwen-Image-Edit-2511** workflows
- Other native ComfyUI **Qwen Image-family** models that still expose the same transformer internals (`img_in`, `txt_norm`, `txt_in`, `transformer_blocks`, `norm_out`, `proj_out`, `time_text_embed`)

Intended scope:

- `Qwen-Image`
- `Qwen-Image-Edit`
- `Qwen-Image-Edit-2511`
- likely other near-identical native Qwen Image variants, as long as ComfyUI keeps the same core transformer layout

## What this does not claim

This repo does **not** claim universal compatibility with:

- Nunchaku / quantized Qwen wrappers
- wrappers that hide or replace the native Qwen transformer core
- ControlNet-heavy runs where per-block residual injection must remain exact
- architectures outside the native Qwen Image-family transformer layout

When unsupported conditions are detected, the node falls back to **real forwards** instead of forcing a broken forecast path.

## Node

### `Spectrum Qwen Model Patcher`

Category: `model/optimization`

Inputs:

- `model`: the ComfyUI `MODEL` to patch
- `warmup_steps`: initial steps forced to run normally
- `tail_actual_steps`: final steps forced to run normally
- `history_points`: number of captured real hidden states used for fitting
- `chebyshev_degree`: polynomial degree for the fit
- `max_consecutive_forecasts`: max skipped Qwen passes in a row before forcing a refresh step
- `ridge_lambda`: ridge regularization strength
- `cache_device`: where hidden-state history is stored (`main_device`, `offload_device`, `cpu`)
- `force_actual_on_control`: force real forwards when control residuals are present
- `debug`: print per-step decisions and a summary

Output:

- patched `MODEL`

## Recommended starting settings

Conservative:

- `warmup_steps = 5`
- `tail_actual_steps = 2`
- `history_points = 5`
- `chebyshev_degree = 3`
- `max_consecutive_forecasts = 1`
- `ridge_lambda = 1e-4`
- `cache_device = main_device`
- `force_actual_on_control = true`
- `debug = false`

More aggressive:

- `warmup_steps = 5`
- `tail_actual_steps = 2`
- `history_points = 6`
- `chebyshev_degree = 3`
- `max_consecutive_forecasts = 2`
- `ridge_lambda = 1e-4`

## How it works

This implementation is deliberately narrow and reviewable.

1. It locates the native Qwen transformer core inside the loaded `MODEL`.
2. On real steps, it runs the original forward path unchanged and captures the hidden state immediately before `norm_out`.
3. It keeps the last `history_points` captured states.
4. On forecast steps, it predicts the next pre-`norm_out` hidden state using a ridge-regularized Chebyshev fit over recent real steps.
5. It then runs the lightweight tail only:
   - `time_text_embed`
   - `norm_out`
   - `proj_out`
6. Warmup, tail, refresh, and unsupported cases stay on the real path.

## Why the forecast target is pre-`norm_out`

For Qwen Image-family models, the expensive part is the long transformer block stack. The `norm_out` + `proj_out` tail is cheap by comparison.

Forecasting the image stream **before** `norm_out` keeps the sampled step-specific modulation in the cheap tail while still bypassing the expensive transformer blocks.

## Installation

### Manual

Clone into your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/xmarre/ComfyUI-Spectrum-Qwen-Proper.git
```

Restart ComfyUI.

### No extra Python dependencies

This repo uses only Python stdlib + whatever ComfyUI already provides.

## Usage

Typical placement:

```text
Load Qwen model -> (optional LoRA/model modifications) -> Spectrum Qwen Model Patcher -> KSampler
```

Important:

- Put Spectrum **after** any node that mutates the model you want sampled.
- Start conservative first.
- Enable `debug` for the first test run so you can confirm the actual/forecast pattern.

## Example workflow pattern

### Qwen-Image-Edit-2511 basic edit

1. Load your Qwen Image Edit model normally
2. Build conditioning with the usual native Qwen Image Edit nodes
3. Insert `Spectrum Qwen Model Patcher` between the loaded model and the sampler
4. Sample with a normal KSampler setup
5. Inspect output quality before pushing forecast aggressiveness harder

## Error handling / fallback behavior

The patcher intentionally prefers correctness over overclaiming support.

It falls back to real forwards when:

- the loaded model is not a supported native Qwen Image-family core
- not enough real history exists yet
- you are still in the warmup region
- you are in the protected tail
- `zero_cond_t` is active
- gradient checkpointing is active
- ControlNet/control residuals are present and `force_actual_on_control = true`
- forecast reconstruction fails for any reason

## Assumptions

This repo assumes the inner Qwen transformer core still exposes fields compatible with the current native Qwen Image-family layout:

- `img_in`
- `txt_norm`
- `txt_in`
- `transformer_blocks`
- `norm_out`
- `proj_out`
- `time_text_embed`

If ComfyUI changes those internals materially, this repo will need updating.

## Caveats and known limitations

1. **This is not a paper-exact universal Qwen port.**
   It is a practical native-ComfyUI Qwen Image-family patcher.

2. **Forecasting target approximation.**
   The implementation forecasts the final pre-`norm_out` image stream. That is the most useful cheap-tail split for this family, but it is still an engineering approximation of the general Spectrum method rather than a claim of paper-exact integration for every Qwen variant.

3. **ControlNet path stays conservative.**
   Per-block control residual injection is not reconstructed analytically here. By default those cases stay on the real path.

4. **Native layout required.**
   Quantized or heavily wrapped variants may expose a different forward path and are not promised here.

5. **No automatic quality oracle.**
   The node does not score image quality for you. You still need to compare results and tune aggression based on your workflow.

6. **CPU cache mode trades VRAM for latency.**
   `cache_device = cpu` can reduce GPU pressure, but it may erase some of the speed win depending on token count and resolution.

## Repository layout

```text
ComfyUI-Spectrum-Qwen-Proper/
├── __init__.py
├── nodes.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── spectrum_qwen/
    ├── __init__.py
    ├── chebyshev.py
    ├── config.py
    ├── constants.py
    ├── controller.py
    ├── forward_qwen.py
    ├── model_introspection.py
    ├── patcher.py
    ├── state.py
    └── utils.py
```

## Development notes

The implementation is intentionally small:

- one node
- one controller
- one forecasting core
- one inner-forward patch

No framework-like layer was added on top.
