# EmbeddingGemma2-Reflex

Public source repository: **EmbeddingGemma2-Reflex**. Weights: [Nurymanau/EmbeddingGemma2-Reflex](https://huggingface.co/Nurymanau/EmbeddingGemma2-Reflex), pinned to `17cfbac9be0f36f051da598a4c7a045e8f249aba` (v0.2.0).


**Research weights for a small text decision model.** Fully fine-tuned EmbeddingGemma 2 text encoder (271M parameters) plus a 238K-parameter choice head. Both independently trained seeds are included, in BF16 and MLX q8. This model chooses among supplied text options; it does not generate explanations or chain-of-thought.

This repository contains the V6 release and a separately evaluated V7 semantic-repair addendum. [Read V7 results and usage](v7/README.md). The root entry points retain V6; V7 heads are explicitly selected. **V6 did not pass our full promotion gate:** novel-rule transfer improved substantially, but new UI accuracy regressed and stable latency parity was not established. These are usable experimental weights, not a claim of general reasoning or superiority over Laya/Jev/Clef.

## Final addendum

The preregistered semantic-repair acceptance checks did not all pass. These heads remain experimental; do not call the regression solved. Both V7 heads and their fresh-test results are published in `v7/`. No general thinking or overall Laya equivalence is claimed.

## What changed

All text encoder weights and the choice head were trained for four epochs. A matched frozen-encoder control used the same data, head initialization, loss and update count. Models were selected on dev before test. Seed17 is the primary release because of its dev ranking, not its later test result. Original base weights were preserved.

The encoder embeds the instruction/state and candidate options separately. A cosine term plus learned gated bilinear residual scores the candidates. No task-label router, rule parser or answer cache is used. Fixed option embeddings can be cached. The same head accepts varying option sets; the training distribution used 3–6 choices.

## Results on one fresh synthetic English test

Live Mac q8 evaluation, 1,552 rows. Transfer is the mean of three groups: new attribute values, new attribute pairings, and both. Many rows share a request with different options; these are not 1,552 independent tasks.

| Model | New values | New pairs | Both new | Transfer mean | UI |
|---|---:|---:|---:|---:|---:|
| Original encoder, cosine | 46.61% | 48.83% | 44.53% | 46.66% | 99.22% |
| Matched frozen, seed17 | 49.48% | 50.78% | 49.22% | 49.83% | 100.00% |
| Full fine-tune, seed17 | **92.71%** | **94.92%** | **95.31%** | **94.31%** | 96.09% |
| Matched frozen, seed41 | 50.78% | 52.73% | 51.56% | 51.69% | 100.00% |
| Full fine-tune, seed41 | 84.11% | 85.16% | 88.67% | 85.98% | 96.09% |
| Earlier frozen V4, 160 epochs | 58.85% | 63.28% | 59.38% | 60.50% | 89.84% |

The four-epoch frozen control is not the best converged frozen baseline. V4 is included as a stronger practical reference with a different training budget. Both full seeds retained 127/127 answers on an older semantic suite, but lost 3.125 percentage points versus the original encoder on the fresh UI split. Test results were not used to reselect checkpoints. GPU and Mac q8 choices differ on 7/1,552 and 8/1,552 rows for the full seeds.

On an M3/16GB Mac, a fixed diagnostic warm benchmark measured p50 **17.6–19.6ms** for Reflex and **17.3–17.4ms** for the installed multilingual Laya MLX reference. Primary repeats were much noisier: Reflex34.4–48.3ms, reference31.5–70.9ms. The full varying-input test measured45.1/49.5ms for the two full seeds. **Do not interpret the fastest warm results as universal latency or proven Laya parity.** All observations and exact scope are in `results.json`.

## Quick start: CPU reference

Python3.12 was used. The CPU reference does not have the reported Mac MLX latency.

```sh
git clone https://github.com/Obscyra-app/EmbeddingGemma2-Reflex.git
cd EmbeddingGemma2-Reflex
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-reference.txt huggingface_hub
python download_weights.py
```

For the Mac Swift runtime, install `requirements-mac.txt` and `huggingface_hub`
instead of the CPU Torch dependencies; the same downloader retrieves both exports.

```python
from reflex import Reflex

model = Reflex("./weights", seed=17)
result = model.predict(
    instructions="Choose the interface command that fulfills the user request.",
    state="Reverse the edit I just made.",
    options=["Print document", "Undo change", "Create folder"],
)
print(result)
model.close()
```

Returned scores use softmax at T=1. **They are not validated calibrated probabilities** for arbitrary inputs. GPU calibration from the experiment is not silently transplanted to the Mac export. Invalid or absent suitable choices can still produce confident wrong answers; no abstention guarantee is implemented.

## Mac / MLX Swift

Install NumPy from `requirements-mac.txt`. The included `swift-runtime` includes the text-only inference source derived from the MIT-licensed EmbeddingGemma2Swift SDK, with the compiled-exact addition. Its public dependencies are MLX0.32.3 and swift-transformers1.3.0. Build with Xcode26.4/Swift6.3:

```bash
cd swift-runtime
xcodebuild -scheme ReflexRuntime -configuration Release -destination 'platform=macOS,arch=arm64' -derivedDataPath build -jobs 2 build
cd ..
```

```python
model = Reflex("./weights", seed=17,
    encoder_executable="./swift-runtime/build/Build/Products/Release/reflex-embed")
```

The `compiled-exact` path was bit-identical to the original SDK on22 dev input sets for each full checkpoint. This does not prove equivalence on every input. Model files in `text-q8` use the SDK's MLX affine8-bit format, not GGUF or a standard Torch quantization format.

## Scope and reproducibility

Training:5,947 examples; dev603; calibration308; one new test1,552. English synthetic attribute selection (including negation and conjunction) plus topic/UI matching. Seeds17 and41; AdamW, encoder LR2e-5, head LR8e-4; effective batch32; FP32 trainable weights with CUDA BF16 autocast. Full training took about24 minutes per seed on one RTX A5000. Total lease, including setup/evaluation/rescue, cost an estimated$0.321 at$0.27/hour, not a provider invoice.

Saved answers were independently rescored:6,208 GPU decisions and12,670 Mac decisions. The initial release includes weights, inference source, aggregate results and SHA manifest. **The private training corpus and full training history are not distributed here; this is not a claim of independently reproducible end-to-end training.**

No broad multilingual benchmark, real UI control, game performance, vision/audio alignment, safety calibration or generative reasoning was established. This text-only derivative must not be substituted for the original multimodal model's aligned text tower without a new alignment evaluation. Do not present this experiment as an official Google, Cloudflare, TypeSafe or Laya project.

## Files and license

Downloaded `weights/seed-17/` and `weights/seed-41/` contain sanitized `head.npz`, BF16 and q8 text exports with tokenizers. `reflex.py` is a standalone NumPy head and CPU/Swift wrapper. No pickle optimizer checkpoints, credentials, private repository history or raw datasets are included.

Apache-2.0, matching the [specific EmbeddingGemma2 base release](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b). Google DeepMind created the base model. See `NOTICE` for modifications and dependencies. SHA-256 checksums in `MODEL_MANIFEST.json` describe the exact release payload.


`SOURCE_MANIFEST.json` records this source export. Model weights stay on Hugging Face; `download_weights.py` verifies their pinned SHA-256 hashes.
