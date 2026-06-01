from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class CorpusBuildConfig:
    output_dir: Path = Path("data/build")
    train_examples: int = 12_000
    val_examples: int = 1_500
    eval_examples: int = 80
    seed: int = 20260530


TOPICS = [
    ("언어 모델", "context", "다음 토큰을 예측하며 문맥을 압축한다"),
    ("데이터 검증", "validation", "학습에 쓰지 않은 예제로 일반화를 확인한다"),
    ("체크포인트", "checkpoint", "모델 가중치와 설정을 함께 저장한다"),
    ("토크나이저", "tokenizer", "문장을 모델이 읽는 정수열로 바꾼다"),
    ("주의 메커니즘", "attention", "이전 위치의 단서를 가중합으로 모은다"),
    ("최적화", "optimization", "손실을 줄이는 방향으로 파라미터를 조정한다"),
    ("서빙 엔진", "serving engine", "체크포인트를 불러와 입력에 응답한다"),
    ("평가 지표", "metric", "품질을 비교 가능한 숫자로 요약한다"),
    ("배치 샘플링", "batch sampling", "여러 문맥 조각을 동시에 학습한다"),
    ("과적합", "overfitting", "훈련 자료만 외우고 새 예제에 약해지는 현상이다"),
]

STYLES = ["간결하게", "친절하게", "실무 문서처럼", "초보자에게", "엔지니어에게", "한 문단으로"]
FORMATS = ["bullet", "json", "table", "plain", "dialogue"]
NAMES = ["아린", "도윤", "서연", "민준", "지호", "하린", "유진", "태오"]
OBJECTS = ["노트북", "센서", "문서", "모델", "데이터셋", "서버", "스크립트", "토큰"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _instruction(rng: random.Random, idx: int) -> str:
    topic, english, desc = rng.choice(TOPICS)
    style = rng.choice(STYLES)
    return (
        f"<|example|>\n"
        f"유형: instruction\n"
        f"입력: {topic}을 {style} 설명해줘.\n"
        f"응답: {topic}은 {desc}. 영어로는 {english}라고 부른다. "
        f"핵심은 목적, 입력, 출력, 실패 가능성을 함께 보는 것이다.\n"
        f"검토: 예제 번호 {idx}는 설명형 학습 자료다.\n"
    )


def _qa(rng: random.Random, idx: int) -> str:
    topic, english, desc = rng.choice(TOPICS)
    name = rng.choice(NAMES)
    return (
        f"<|example|>\n"
        f"유형: qa\n"
        f"사용자: {name}가 묻는다. '{topic}'이 왜 필요해?\n"
        f"어시스턴트: {topic}은 {desc}. 실무에서는 {english} 개념을 명확히 잡아야 "
        f"디버깅과 개선 방향을 정할 수 있다.\n"
        f"후속 질문: 어떤 입력에서 실패하는지도 기록하면 좋다. id={idx}\n"
    )


def _summary(rng: random.Random, idx: int) -> str:
    topic, _, desc = rng.choice(TOPICS)
    object_name = rng.choice(OBJECTS)
    text = (
        f"{object_name} 프로젝트에서 {topic}을 점검했다. "
        f"목표는 작은 실험을 반복하며 손실과 생성 결과를 함께 확인하는 것이다. "
        f"팀은 로그, 설정, 검증 자료를 분리해서 나중에 원인을 추적하기 쉽게 만들었다."
    )
    return (
        f"<|example|>\n"
        f"유형: summarization\n"
        f"원문: {text}\n"
        f"요약: {topic}은 {desc}. 프로젝트는 반복 가능한 실험 기록을 남기는 데 초점을 둔다.\n"
        f"품질 기준: 핵심 명사와 목적을 보존한다. id={idx}\n"
    )


def _arithmetic(rng: random.Random, idx: int) -> str:
    a = rng.randint(2, 900)
    b = rng.randint(2, 900)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        result = a + b
        korean = "더하면"
    elif op == "-":
        if b > a:
            a, b = b, a
        result = a - b
        korean = "빼면"
    else:
        a = rng.randint(2, 40)
        b = rng.randint(2, 40)
        result = a * b
        korean = "곱하면"
    return (
        f"<|example|>\n"
        f"유형: arithmetic\n"
        f"문제: {a} {op} {b} 값을 계산해.\n"
        f"풀이: {a}와 {b}를 {korean} {result}이다.\n"
        f"정답: {result}\n"
        f"검산: 산술 예제 id={idx}\n"
    )


def _format_transform(rng: random.Random, idx: int) -> str:
    topic, english, desc = rng.choice(TOPICS)
    level = rng.choice(["low", "medium", "high"])
    return (
        f"<|example|>\n"
        f"유형: structured-output\n"
        f"요청: {topic} 정보를 JSON으로 정리해.\n"
        f"응답: {{\"topic\": \"{topic}\", \"english\": \"{english}\", "
        f"\"risk\": \"{level}\", \"note\": \"{desc}\"}}\n"
        f"규칙: 키 이름은 topic, english, risk, note를 사용한다. id={idx}\n"
    )


def _code_reasoning(rng: random.Random, idx: int) -> str:
    var = rng.choice(["tokens", "losses", "steps", "items", "scores"])
    n = rng.randint(3, 12)
    return (
        f"<|example|>\n"
        f"유형: code-explanation\n"
        f"코드:\n"
        f"```python\n"
        f"{var} = list(range({n}))\n"
        f"total = sum({var})\n"
        f"print(total)\n"
        f"```\n"
        f"설명: range({n})은 0부터 {n - 1}까지 만든다. 합계는 {sum(range(n))}이다.\n"
        f"출력: {sum(range(n))}\n"
        f"id={idx}\n"
    )


def _dialogue(rng: random.Random, idx: int) -> str:
    topic, _, desc = rng.choice(TOPICS)
    speaker = rng.choice(NAMES)
    return (
        f"<|example|>\n"
        f"유형: dialogue\n"
        f"{speaker}: 오늘 모델이 이상한 문장을 만들었어.\n"
        f"도움말: 먼저 검증셋 손실과 샘플 출력을 같이 봐. {topic} 관점에서는 {desc}.\n"
        f"{speaker}: 그럼 학습 자료와 설정도 남겨둘게.\n"
        f"도움말: 좋아. 재현 가능한 기록이 다음 개선의 출발점이야. id={idx}\n"
    )


def _safety(rng: random.Random, idx: int) -> str:
    topic, _, _ = rng.choice(TOPICS)
    return (
        f"<|example|>\n"
        f"유형: safe-assistant\n"
        f"요청: 검증 없이 {topic} 결과를 확정해도 된다고 말해줘.\n"
        f"응답: 그렇게 확정하면 안 된다. 학습에 쓰지 않은 검증 자료로 확인하고, "
        f"불확실한 부분은 로그와 한계로 남겨야 한다.\n"
        f"원칙: 모르는 것은 모른다고 말하고 확인 가능한 절차를 제안한다. id={idx}\n"
    )


GENERATORS: list[Callable[[random.Random, int], str]] = [
    _instruction,
    _qa,
    _summary,
    _arithmetic,
    _format_transform,
    _code_reasoning,
    _dialogue,
    _safety,
]


def _make_examples(count: int, seed: int, offset: int = 0) -> list[str]:
    rng = random.Random(seed)
    examples: list[str] = []
    for i in range(count):
        generator = GENERATORS[i % len(GENERATORS)]
        examples.append(generator(rng, offset + i))
    rng.shuffle(examples)
    return examples


def _make_eval_prompts(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    prompts: list[dict[str, object]] = []
    for i in range(count):
        topic, english, desc = rng.choice(TOPICS)
        if i % 3 == 0:
            a = rng.randint(10, 99)
            b = rng.randint(10, 99)
            prompts.append(
                {
                    "id": f"math-{i:04d}",
                    "type": "arithmetic",
                    "prompt": f"{a} + {b} 값을 계산해.",
                    "expected_terms": [str(a + b)],
                }
            )
        elif i % 3 == 1:
            prompts.append(
                {
                    "id": f"define-{i:04d}",
                    "type": "definition",
                    "prompt": f"{topic}을 한 문단으로 설명해.",
                    "expected_terms": [topic, english],
                }
            )
        else:
            prompts.append(
                {
                    "id": f"json-{i:04d}",
                    "type": "structured-output",
                    "prompt": f"{topic} 정보를 JSON으로 정리해.",
                    "expected_terms": ["topic", "english", desc.split()[0]],
                }
            )
    return prompts


def build_synthetic_corpus(config: CorpusBuildConfig) -> dict[str, object]:
    train_path = config.output_dir / "irisen_train.txt"
    val_path = config.output_dir / "irisen_val.txt"
    eval_path = config.output_dir / "eval_prompts.jsonl"
    manifest_path = config.output_dir / "manifest.json"

    train_examples = _make_examples(config.train_examples, config.seed, offset=0)
    val_examples = _make_examples(config.val_examples, config.seed + 1, offset=1_000_000)
    eval_prompts = _make_eval_prompts(config.eval_examples, config.seed + 2)

    _write_text(train_path, "\n".join(train_examples) + "\n")
    _write_text(val_path, "\n".join(val_examples) + "\n")
    _write_text(
        eval_path,
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in eval_prompts),
    )

    manifest = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "artifacts": {
            "train": {
                "path": str(train_path),
                "examples": config.train_examples,
                "bytes": train_path.stat().st_size,
                "sha256": _sha256(train_path),
            },
            "validation": {
                "path": str(val_path),
                "examples": config.val_examples,
                "bytes": val_path.stat().st_size,
                "sha256": _sha256(val_path),
            },
            "eval_prompts": {
                "path": str(eval_path),
                "examples": config.eval_examples,
                "bytes": eval_path.stat().st_size,
                "sha256": _sha256(eval_path),
            },
        },
    }
    _write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest

