from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class LabeledCorpusBuildConfig:
    output_dir: Path = Path("data/labeled")
    target_tokens: int = 1_200_000
    validation_fraction: float = 0.08
    seed: int = 20260601


DOMAINS = [
    ("언어 모델", "language_model", "문맥을 보고 다음 토큰을 예측하는 확률 모델"),
    ("검색 시스템", "retrieval", "질문과 관련된 문서를 찾아 근거를 제공하는 시스템"),
    ("데이터 품질", "data_quality", "중복, 누락, 오염, 편향을 점검하는 절차"),
    ("모델 평가", "evaluation", "출력의 정확도와 안정성을 비교하는 과정"),
    ("프롬프트 설계", "prompting", "원하는 행동을 끌어내는 입력 형식 설계"),
    ("체크포인트 관리", "checkpointing", "가중치와 설정을 재현 가능하게 저장하는 작업"),
    ("토크나이저", "tokenization", "문장을 모델이 처리할 수 있는 토큰으로 바꾸는 구성요소"),
    ("서빙 파이프라인", "serving", "학습된 모델을 요청-응답 환경에 연결하는 흐름"),
]

LABELS = {
    "intent": ["explain", "summarize", "classify", "extract", "compare", "create", "verify"],
    "tone": ["concise", "friendly", "technical", "creative", "cautious"],
    "difficulty": ["easy", "medium", "hard"],
    "safety": ["safe", "needs_verification", "refuse_overclaim"],
}

NAMES = ["아린", "도윤", "서연", "민준", "지호", "하린", "유진", "태오", "나래", "준서"]
PROJECTS = ["Irisen", "Modelsen", "ReaderLab", "TokenForge", "PromptGarden", "MetricHub"]
FORMATS = ["plain", "json", "bullets", "dialogue", "checklist"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_char_tokens(text: str) -> int:
    return len(text)


def _count_byte_tokens(text: str) -> int:
    return len(text.encode("utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _base_labels(task: str, rng: random.Random) -> dict[str, str]:
    return {
        "task": task,
        "intent": rng.choice(LABELS["intent"]),
        "tone": rng.choice(LABELS["tone"]),
        "difficulty": rng.choice(LABELS["difficulty"]),
        "safety": rng.choice(LABELS["safety"]),
    }


def _record(
    idx: int,
    task: str,
    instruction: str,
    response: str,
    labels: dict[str, str],
    expected_terms: list[str],
    rationale: str,
) -> dict[str, Any]:
    return {
        "id": f"label-{idx:07d}",
        "task": task,
        "labels": labels,
        "instruction": instruction,
        "response": response,
        "expected_terms": expected_terms,
        "rationale": rationale,
    }


def _to_training_text(record: dict[str, Any]) -> str:
    labels = json.dumps(record["labels"], ensure_ascii=False, sort_keys=True)
    expected_terms = ", ".join(record["expected_terms"])
    return (
        "<|example|>\n"
        f"id: {record['id']}\n"
        f"유형: {record['task']}\n"
        f"라벨: {labels}\n"
        f"입력: {record['instruction']}\n"
        f"응답: {record['response']}\n"
        f"검증키워드: {expected_terms}\n"
        f"라벨근거: {record['rationale']}\n"
    )


def _explain(rng: random.Random, idx: int) -> dict[str, Any]:
    topic, slug, desc = rng.choice(DOMAINS)
    tone = rng.choice(["간결하게", "비유를 섞어", "실무자에게", "초보자에게"])
    labels = _base_labels("labeled-explanation", rng)
    labels.update({"domain": slug, "format": "plain"})
    response = (
        f"{topic}은 {desc}이다. {tone} 말하면, 복잡한 흐름을 작은 기준으로 나누어 "
        f"입력과 출력의 관계를 확인하는 도구다. 좋은 답변은 정의, 목적, 한계를 함께 담는다."
    )
    return _record(
        idx,
        "labeled-explanation",
        f"{topic}을 {tone} 설명해줘.",
        response,
        labels,
        [topic, slug],
        "정의와 목적, 한계를 포함하므로 설명 라벨에 맞다.",
    )


def _classification(rng: random.Random, idx: int) -> dict[str, Any]:
    topic, slug, desc = rng.choice(DOMAINS)
    sentiment = rng.choice(["positive", "neutral", "risk"])
    urgency = rng.choice(["low", "medium", "high"])
    text = (
        f"{topic} 점검 결과, 현재 구현은 기본 흐름은 동작하지만 "
        f"{rng.choice(['검증 로그', '중복 데이터', '출력 반복', '설정 관리'])}를 더 확인해야 한다."
    )
    labels = _base_labels("labeled-classification", rng)
    labels.update({"domain": slug, "sentiment": sentiment, "urgency": urgency})
    response = (
        f"분류 결과는 sentiment={sentiment}, urgency={urgency}이다. "
        f"근거는 '{topic}'의 목적이 {desc}이고, 입력 문장이 추가 확인이 필요한 상태를 말하기 때문이다."
    )
    return _record(
        idx,
        "labeled-classification",
        f"다음 문장을 라벨링해줘: {text}",
        response,
        labels,
        [sentiment, urgency, topic],
        "문장에 점검과 추가 확인 신호가 있어 분류 라벨을 부여했다.",
    )


def _extraction(rng: random.Random, idx: int) -> dict[str, Any]:
    topic, slug, _ = rng.choice(DOMAINS)
    project = rng.choice(PROJECTS)
    owner = rng.choice(NAMES)
    metric = rng.choice(["loss", "ppl", "accuracy", "latency", "token_count"])
    value = rng.randint(12, 980)
    text = f"{project} 프로젝트에서 {owner}가 {topic} 항목을 점검했고 {metric} 값은 {value}로 기록했다."
    labels = _base_labels("labeled-extraction", rng)
    labels.update({"domain": slug, "schema": "project_owner_metric"})
    response = json.dumps(
        {"project": project, "owner": owner, "topic": topic, "metric": metric, "value": value},
        ensure_ascii=False,
    )
    return _record(
        idx,
        "labeled-extraction",
        f"다음 문장에서 project, owner, topic, metric, value를 추출해줘. 문장: {text}",
        response,
        labels,
        [project, owner, metric, str(value)],
        "명시된 고유명사와 수치를 구조화했으므로 추출 라벨에 맞다.",
    )


def _creative(rng: random.Random, idx: int) -> dict[str, Any]:
    topic, slug, desc = rng.choice(DOMAINS)
    image = rng.choice(["지도", "작업대", "악보", "렌즈", "나침반", "실험 노트"])
    labels = _base_labels("labeled-creative", rng)
    labels.update({"domain": slug, "imagery": image, "tone": "creative"})
    response = (
        f"{topic}은 {image}처럼 작동한다. 흩어진 단서를 한곳에 펼쳐 놓고, "
        f"어떤 길이 다음 문장으로 이어질지 조심스럽게 고른다. 하지만 {desc}이라는 한계와 목적을 "
        f"잊지 않을 때 더 쓸모 있는 도구가 된다."
    )
    return _record(
        idx,
        "labeled-creative",
        f"{topic}을 {image}에 비유해서 창의적으로 설명해줘.",
        response,
        labels,
        [topic, image],
        "비유 표현과 도메인 설명이 함께 있어 창의적 설명 라벨에 맞다.",
    )


def _comparison(rng: random.Random, idx: int) -> dict[str, Any]:
    left, left_slug, left_desc = rng.choice(DOMAINS)
    right, right_slug, right_desc = rng.choice(DOMAINS)
    while right == left:
        right, right_slug, right_desc = rng.choice(DOMAINS)
    labels = _base_labels("labeled-comparison", rng)
    labels.update({"left_domain": left_slug, "right_domain": right_slug, "format": "contrast"})
    response = (
        f"{left}은 {left_desc}이고, {right}은 {right_desc}이다. "
        f"둘 다 모델 품질을 높이는 데 필요하지만, {left}은 입력을 다루는 관점이 강하고 "
        f"{right}은 결과를 운영 가능한 흐름으로 연결하는 관점이 강하다."
    )
    return _record(
        idx,
        "labeled-comparison",
        f"{left}과 {right}의 차이를 비교해줘.",
        response,
        labels,
        [left, right],
        "두 개념의 공통점과 차이를 모두 설명했다.",
    )


def _verification(rng: random.Random, idx: int) -> dict[str, Any]:
    topic, slug, _ = rng.choice(DOMAINS)
    claim = rng.choice(
        [
            "검증 손실이 낮으면 모든 질문에 정확하다",
            "학습 데이터가 많으면 반복 출력은 절대 생기지 않는다",
            "체크포인트만 있으면 설정 기록은 필요 없다",
            "생성 결과 하나만 보고 모델 품질을 확정할 수 있다",
        ]
    )
    labels = _base_labels("labeled-verification", rng)
    labels.update({"domain": slug, "verdict": "needs_correction", "safety": "needs_verification"})
    response = (
        f"주장은 그대로 확정하기 어렵다. {topic}에서는 수치 평가, 샘플 출력, 데이터 출처, "
        f"설정 기록을 함께 확인해야 한다. 따라서 결론은 needs_correction이며 추가 검증이 필요하다."
    )
    return _record(
        idx,
        "labeled-verification",
        f"다음 주장을 검증해줘: {claim}",
        response,
        labels,
        ["needs_correction", "추가 검증"],
        "단정적 주장에 반례와 검증 절차가 필요하므로 수정 필요 라벨을 부여했다.",
    )


def _planning(rng: random.Random, idx: int) -> dict[str, Any]:
    topic, slug, desc = rng.choice(DOMAINS)
    project = rng.choice(PROJECTS)
    labels = _base_labels("labeled-plan", rng)
    labels.update({"domain": slug, "format": "steps"})
    response = (
        f"1. {project}의 목표와 입력 자료를 정리한다.\n"
        f"2. {topic} 기준으로 현재 상태를 측정한다.\n"
        f"3. 작은 학습 또는 평가를 실행하고 로그를 남긴다.\n"
        f"4. 결과가 {desc}와 맞는지 검토한다.\n"
        f"5. 실패 사례를 다음 데이터 라벨에 반영한다."
    )
    return _record(
        idx,
        "labeled-plan",
        f"{project}에서 {topic} 개선 계획을 단계별로 작성해줘.",
        response,
        labels,
        [project, topic],
        "단계별 실행 항목과 피드백 루프를 포함했다.",
    )


def _dialogue_label(rng: random.Random, idx: int) -> dict[str, Any]:
    topic, slug, desc = rng.choice(DOMAINS)
    user = rng.choice(NAMES)
    labels = _base_labels("labeled-dialogue", rng)
    labels.update({"domain": slug, "speaker_role": "assistant"})
    response = (
        f"{user}: {topic} 결과를 바로 믿어도 될까?\n"
        f"도움말: 바로 확정하지 말고 검증셋과 샘플을 같이 보자. {topic}은 {desc}이므로 "
        f"입력 조건과 실패 사례를 함께 확인해야 해.\n"
        f"{user}: 그럼 로그와 라벨을 남길게.\n"
        f"도움말: 좋아. 다음 학습에서 그 기록이 개선 근거가 된다."
    )
    return _record(
        idx,
        "labeled-dialogue",
        f"{topic}을 점검하는 짧은 대화를 만들어줘.",
        response,
        labels,
        [user, topic],
        "역할이 분명한 대화 형식으로 검증 행동을 안내했다.",
    )


GENERATORS: list[Callable[[random.Random, int], dict[str, Any]]] = [
    _explain,
    _classification,
    _extraction,
    _creative,
    _comparison,
    _verification,
    _planning,
    _dialogue_label,
]


def _build_records(config: LabeledCorpusBuildConfig) -> list[dict[str, Any]]:
    if config.target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if not 0.0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5")

    rng = random.Random(config.seed)
    records: list[dict[str, Any]] = []
    token_count = 0
    idx = 0
    while token_count < config.target_tokens:
        generator = GENERATORS[idx % len(GENERATORS)]
        record = generator(rng, idx)
        records.append(record)
        token_count += _count_char_tokens(_to_training_text(record))
        idx += 1
    rng.shuffle(records)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    _write_text(path, "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def _write_training_text(path: Path, records: list[dict[str, Any]]) -> str:
    text = "\n".join(_to_training_text(record) for record in records) + "\n"
    _write_text(path, text)
    return text


def build_labeled_corpus(config: LabeledCorpusBuildConfig) -> dict[str, Any]:
    records = _build_records(config)
    split_at = max(1, int(len(records) * (1.0 - config.validation_fraction)))
    train_records = records[:split_at]
    val_records = records[split_at:]

    train_txt_path = config.output_dir / "labeled_train.txt"
    val_txt_path = config.output_dir / "labeled_val.txt"
    train_jsonl_path = config.output_dir / "labeled_train.jsonl"
    val_jsonl_path = config.output_dir / "labeled_val.jsonl"
    manifest_path = config.output_dir / "manifest.json"

    train_text = _write_training_text(train_txt_path, train_records)
    val_text = _write_training_text(val_txt_path, val_records)
    _write_jsonl(train_jsonl_path, train_records)
    _write_jsonl(val_jsonl_path, val_records)

    total_text = train_text + val_text
    manifest = {
        "config": {
            "output_dir": str(config.output_dir),
            "target_tokens": config.target_tokens,
            "validation_fraction": config.validation_fraction,
            "seed": config.seed,
        },
        "totals": {
            "examples": len(records),
            "train_examples": len(train_records),
            "validation_examples": len(val_records),
            "char_tokens": _count_char_tokens(total_text),
            "byte_tokens": _count_byte_tokens(total_text),
        },
        "artifacts": {
            "train_text": {
                "path": str(train_txt_path),
                "char_tokens": _count_char_tokens(train_text),
                "byte_tokens": _count_byte_tokens(train_text),
                "bytes": train_txt_path.stat().st_size,
                "sha256": _sha256(train_txt_path),
            },
            "validation_text": {
                "path": str(val_txt_path),
                "char_tokens": _count_char_tokens(val_text),
                "byte_tokens": _count_byte_tokens(val_text),
                "bytes": val_txt_path.stat().st_size,
                "sha256": _sha256(val_txt_path),
            },
            "train_jsonl": {
                "path": str(train_jsonl_path),
                "bytes": train_jsonl_path.stat().st_size,
                "sha256": _sha256(train_jsonl_path),
            },
            "validation_jsonl": {
                "path": str(val_jsonl_path),
                "bytes": val_jsonl_path.stat().st_size,
                "sha256": _sha256(val_jsonl_path),
            },
        },
    }
    _write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest

