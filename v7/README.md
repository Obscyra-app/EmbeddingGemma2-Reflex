# V7 semantic repair — v0.2.0

The preregistered semantic-repair acceptance checks did not all pass. These heads remain experimental; do not call the regression solved.

V6 encoder weights are unchanged. Only the 238,017-parameter head was repaired locally with balanced new semantic examples and rule distillation. Both seeds and epoch-zero baselines were retained; epoch14/seed17 and epoch4/seed41 were selected on dev before test. **Seed41 is the dev-selected primary**, regardless of which seed later scores higher on test.

## A separate fresh test, 1,679 rows

Do not compare these percentages directly to the V6 table: requests, held-out attribute values and semantic paraphrases changed. V6 and V7 below were tested on identical new inputs. The logical families and known-value inventory generator remain the same; these are not new logical laws. Many rows share a request with different option sets. The new semantic test has50 distinct request sentences, and is not evidence of broad language reasoning.

| Variant | Transfer mean | UI | Topic |
|---|---:|---:|---:|
| raw | 47.66% | 96.08% | 100.00% |
| v6-s17 | 95.01% | 95.29% | 100.00% |
| v7-s17 | 95.23% | 99.61% | 100.00% |
| v6-s41 | 83.94% | 91.76% | 100.00% |
| v7-s41 | 83.94% | 99.61% | 100.00% |

The old V6 UI regression suite is checked separately; it did not choose checkpoints. Seed17: 128/128. Seed41: 124/128. Original encoder: 127/128. Older semantic retention: 127/127 and 124/127.

## Use the repaired head with its matching encoder

Build the same Swift runtime described in the root README. The V7 accuracy measurements use MLX q8; BF16 CPU V7 equivalence has not been established.

```python
from reflex import Reflex
model = Reflex("./weights", seed=41,
    encoder_executable="./swift-runtime/build/Build/Products/Release/reflex-embed",
    head_path="./weights/v7/seed-41/head.npz")
print(model.predict("Choose the most suitable command.", "Reverse the last edit.",
                    ["Undo change", "Print document", "Create folder"]))
model.close()
```

The root heads remain V6 to preserve the original entry points. The immutable tag `v0.1.0` preserves the complete first release. Do not mix the seed17 head with the seed41 encoder.

## Latency and limits

Two predeclared matched warm runs (20 dev cases x3 shuffled passes) are recorded in `results.json`, including every observed result. Static option cache is allowed; query and answer caching are not. The reference is the installed multilingual Laya MLX implementation, whose parity with upstream Torch was not established. Earlier latency variability remains relevant. Passing a narrow timing check would not establish universal model parity.

The model returns choices, not generated reasoning. Results cover synthetic English attribute/topic/UI tasks. Training data are not distributed; this is a weights and inference release, not complete training reproduction. No new cloud rental was needed for V7.
