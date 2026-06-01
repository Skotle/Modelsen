# irisen-model

상용 LLM 저장소의 기본 형태를 축소해서 구현한 byte-level causal language model입니다. 교육용 장난감에 머물지 않도록 모델 정의, 토크나이저, 데이터 로딩, trainer, checkpoint schema, generation sampler, serving wrapper, CLI를 계층별로 분리했습니다.

## 빠른 실행

```powershell
python -m unittest
python scripts/build_corpus.py --out-dir data/build --train-examples 12000 --val-examples 1500 --eval-examples 80
python -m irisen_model.train --config configs/irisen-local-build.json --data data/build/irisen_train.txt --val-data data/build/irisen_val.txt
python scripts/evaluate.py --checkpoint runs/irisen_tiny.pt --data data/build/irisen_val.txt
python -m irisen_model.generate --checkpoint runs/irisen_tiny.pt --prompt "언어 모델은" --tokens 120
```

## 클라우드 빌드

클라우드 runner용 설정을 포함합니다.

- GitHub Actions: `.github/workflows/cloud-build.yml`
- Google Cloud Build: `cloudbuild.yaml`
- AWS CodeBuild: `buildspec.yml`
- 컨테이너 빌드: `Dockerfile`

자세한 내용은 [docs/cloud-build.md](docs/cloud-build.md)를 보세요.

스크립트 진입점도 제공합니다.

```powershell
python scripts/train.py --config configs/irisen-tiny.json
python scripts/generate.py --checkpoint runs/irisen_tiny.pt --prompt "언어 모델은"
```

## 데이터 빌드

`scripts/build_corpus.py`는 외부 다운로드 없이 재현 가능한 학습/검증 자료를 만듭니다.

- `data/build/irisen_train.txt`: instruction, Q/A, 요약, 산술, JSON 변환, 코드 설명, 대화, 안전 응답 예제
- `data/build/irisen_val.txt`: 학습 파일과 분리된 검증 corpus
- `data/build/eval_prompts.jsonl`: generation 평가용 prompt와 기대 키워드
- `data/build/manifest.json`: 파일 크기, 예제 수, sha256

## 소스 구조

```text
irisen_model/
  modeling/       IrisenConfig, Transformer block, IrisenForCausalLM
  tokenization/   UTF-8 byte tokenizer
  data/           text corpus loading, synthetic corpus build, batch sampling
  training/       TrainConfig, Trainer, metrics
  generation/     top-k sampling and autoregressive token generation
  checkpoints/    checkpoint save/load schema
  serving/        TextGenerationEngine for inference-style use
  cli/            train/generate command implementations
configs/          experiment JSON configs
scripts/          direct script entrypoints
tests/            unit and smoke tests
docs/             architecture notes
```

기존 import도 유지됩니다.

```python
from irisen_model import ByteTokenizer, IrisenConfig, IrisenForCausalLM
from irisen_model import LanguageModel, ModelConfig  # compatibility aliases
```

## 더 크게 학습하기

`data/tiny_korean.txt` 대신 직접 만든 `.txt` 파일을 넣고 config나 CLI override를 키우면 됩니다.

```powershell
python -m irisen_model.train --data data/my_corpus.txt --steps 2000 --context-size 128 --d-model 128 --layers 4 --heads 4
```

이 코드는 상용 모델의 소스 배치를 닮은 compact reference입니다. 실제 상용급 품질에는 대규모 데이터, BPE/SentencePiece tokenizer, mixed precision, distributed training, 평가 harness, model registry, safety layer, monitoring이 추가로 필요합니다.
