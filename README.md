# Verifier-Guided Diffusion Plan Repair

Agent가 생성한 실행 계획(plan)의 구조적 오류를 validator로 진단하고, 이후 단계에서 복구기(결정론/AR/diffusion)를
동일 인터페이스로 붙여 비교하는 연구용 실험 레포.

현재 범위는 **Ticket 001** — 평가 하네스의 첫 조각인 닫힌 루프다.

```
reference plan → corruption injection → broken plan → validator → (injected ↔ detected error 비교)
```

## 구성

| 패키지 | 역할 |
|---|---|
| `plan_repair.schema` | `AgentTask` / `AgentPlan` / `Step`, corruption 메타데이터(`InjectedError`, `CorruptionResult`) |
| `plan_repair.canonical` | plan 정규화 — 결정론적 canonical JSON + step별 해시 |
| `plan_repair.validation` | 핵심 validator 5종 (schema, tool existence, dependency existence, DAG cycle, ordering) |
| `plan_repair.corruption` | corruption injector 2종 (broken_dependency, step_deletion) |
| `plan_repair.data` | reference plan 데이터 (도메인 B: 20-step 데이터 분석 파이프라인) |
| `plan_repair.repair` | [B 단계 예비] 복구기 port 자리 — 비어 있음 |

## 개발

```bash
uv sync            # 3.12 가상환경 + 의존성
uv run pytest      # 테스트
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## 문서

- `diffusion-plan-repair-plan.md` — 프로젝트 전체 계획
- `ticket-001-schema-validator-corruption.md` — 현재 티켓
- `poc-scenario-design.md` — 도메인 시나리오 + corruption golden
- `project-setup-standard.md` — 고정 도구 체인·레이아웃·규약
- `adr-index.md` — 설계 결정 기록
