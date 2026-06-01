# Architecture

`irisen-model`은 decoder-only Transformer를 상용 LLM 코드베이스와 비슷한 계층으로 나눈 compact implementation입니다.

## Layers

- `modeling`: 모델 config와 PyTorch module만 둡니다. `IrisenForCausalLM`은 logits/loss를 반환하고, generation 구현 세부사항은 `generation`으로 위임합니다.
- `tokenization`: tokenizer contract를 담당합니다. 현재는 UTF-8 byte tokenizer라 vocab artifact가 필요 없습니다.
- `data`: 텍스트 파일을 token tensor로 변환하고 random causal LM batch를 만듭니다. `synthetic_corpus`는 외부 의존성 없이 train/validation/eval prompt 자료를 재현 가능하게 생성합니다.
- `training`: `Trainer`가 optimizer, evaluation, checkpoint save policy를 소유합니다.
- `generation`: sampling primitive입니다. serving, CLI, model compatibility method가 공통으로 사용합니다.
- `checkpoints`: checkpoint schema version, model config, tokenizer metadata, train config, state dict를 저장합니다.
- `serving`: checkpoint를 로드하고 문자열 prompt를 받아 문자열 응답을 내는 inference wrapper입니다.
- `cli`: argparse 진입점만 둡니다. 실제 학습/추론 로직은 하위 계층으로 넘깁니다.

## Checkpoint Contents

```text
schema_version
model_type
model_config
tokenizer
train_config
model_state
step
loss
```

이 구분을 두면 나중에 tokenizer 교체, model registry, distributed trainer, HTTP serving layer를 붙일 때 기존 모델 코어를 흔들지 않고 확장할 수 있습니다.

## Data Artifacts

`scripts/build_corpus.py`가 만드는 기본 산출물은 다음과 같습니다.

```text
data/build/irisen_train.txt
data/build/irisen_val.txt
data/build/eval_prompts.jsonl
data/build/manifest.json
```

검증 파일은 trainer의 `--val-data` 인자로 별도 로드할 수 있으므로, train split에서 validation 예제가 섞이는 일을 피할 수 있습니다.
