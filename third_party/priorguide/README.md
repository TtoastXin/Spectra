# Vendored PriorGuide code

The files under `sim/` are copied from PriorGuide
(<https://github.com/acerbilab/prior-guide>), commit
`b4852fc1a3c37eeec71affa0078874e1977d8cbd`, directory `priorg/sim/`. They are
distributed under the MIT License reproduced in `LICENSE` in this directory.

They provide the Simformer network, its denoising score-matching loss and the
structured conditioning masks used in training, the C2ST and MMTV metrics, and
the constants that standardize the BCI (`bav`) responses, so that no separate
PriorGuide checkout is needed.

## Files

| File | Upstream path | Status |
|---|---|---|
| `sim/__init__.py` | `priorg/sim/__init__.py` | unchanged (empty upstream) |
| `sim/nn/__init__.py` | `priorg/sim/nn/__init__.py` | **replaced by an empty file** |
| `sim/nn/transformers.py` | `priorg/sim/nn/transformers.py` | unchanged |
| `sim/nn/attention.py` | `priorg/sim/nn/attention.py` | unchanged |
| `sim/nn/tokenizer.py` | `priorg/sim/nn/tokenizer.py` | unchanged |
| `sim/nn/helpers.py` | `priorg/sim/nn/helpers.py` | unchanged |
| `sim/nn/loss_fn.py` | `priorg/sim/nn/loss_fn.py` | unchanged |
| `sim/core/__init__.py` | `priorg/sim/core/__init__.py` | **replaced by an empty file** |
| `sim/core/custom_primitives/__init__.py` | `priorg/sim/core/custom_primitives/__init__.py` | unchanged (empty upstream) |
| `sim/core/custom_primitives/custom_inverse.py` | `priorg/sim/core/custom_primitives/custom_inverse.py` | unchanged |
| `sim/methods/__init__.py` | `priorg/sim/methods/__init__.py` | unchanged (empty upstream) |
| `sim/methods/metrics.py` | `priorg/sim/methods/metrics.py` | unchanged |
| `sim/tasks/__init__.py` | `priorg/sim/tasks/__init__.py` | unchanged (empty upstream) |
| `sim/tasks/bav.py` | `priorg/sim/tasks/bav.py` | unchanged |
| `sim/utils/__init__.py` | `priorg/sim/utils/__init__.py` | unchanged |
| `sim/utils/conditional_mask.py` | `priorg/sim/utils/conditional_mask.py` | unchanged |

## Modifications

The upstream `sim/nn/__init__.py` and `sim/core/__init__.py` import every
module of their packages, including autoregressive and coupling layers, U-Nets
and the `sim.core` transformation utilities. None of these modules is used
here, and importing them would require additional dependencies such as
IPython. Both files are therefore empty in this copy. Neither upstream file
changes any global JAX configuration, so leaving out these imports does not
change any computation. Every other file is unchanged from the upstream
commit.
