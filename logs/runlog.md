# N2P run log (append one line per run)

Format: `## [YYYY-MM-DD] <script> | model=<m> task=<t> seed=<s> | <one-line result> | results/<path>`

## [2026-06-14] scaffold | repo initialized | Version A, week-1 number-rep + circuit-sanity scripts written (untested on GPU) | —
## [2026-06-14] revise | circuit discovery ACDC→Edge Pruning+Tracr; Llama build_layers prior (16,21); tasks=clean-core+stress-set; helix intercept bug fixed (synthetic R²=1.0) | logic unit-checked, GPU-untested | —
## [2026-06-14] add | week1_accuracy_probe (frozen few-shot accuracy per task×framing); per-task framings; greater_than canonical template; gated-MLP caveat | logic unit-checked, GPU-untested | —
## [2026-06-16] revise | run_helix_fit.py + helix.py | week1 helix fit aligned to kantamneni2025: --hi default 360->99 (below 3-digit discontinuity); poly baseline -> [a..a^(2k+1)]+intercept (capacity-matched to helix 2k+1+intercept, was [a^0..a^(2k)]); added --context {bare,addition} to fit helix(a) on operand-a token in "{a}+{b}=" prompts (paper §4.3). Logic unit-checked offline (helix R2=1.00, poly=0.99 on synthetic helix; shuffled-a=0.02). GPU-untested | —
