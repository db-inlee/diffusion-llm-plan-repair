# Verifier-Guided Diffusion Plan Repair

Agent가 생성한 실행 계획(plan)의 구조적 오류를 validator로 진단하고, diffusion LLM으로 오류 구간만 선택적으로 복구하는 연구용 실험 레포. diffusion이 "정상 단계를 덜 망치면서" 복구하는지를 측정하는 것이 목표다.

> **상태:** A단계(평가 하네스) 완료. B단계(복구기 연결)는 예정. 아래 "진행 상황" 참고.

---

## 무엇을, 왜

**문제.** Agent가 세운 실행 계획은 자주 망가진다 — 단계가 빠지거나, 존재하지 않는 단계를 참조하거나, 순서가 꼬이거나, 잘못된 도구를 쓴다. 이를 고치는 방법은 두 가지다. (1) 계획 전체를 다시 생성하거나, (2) 망가진 구간만 고치거나. 전자는 멀쩡한 단계까지 위험에 빠뜨린다.

**접근.** diffusion LLM은 계획 전체를 양방향으로 보면서 특정 구간만 다시 채울 수 있다(selective remask). 이 프로젝트는 **validator가 오류 위치를 좁혀 주면, diffusion이 그 구간을 스스로 진단·수정**하는 분업을 검증한다. validator는 "어디가 틀렸나"까지, diffusion은 "무엇을 어떻게 고칠까"를 맡는다.

**정직한 가설.** diffusion의 실제 경쟁자는 "이 단계만 고쳐줘"라는 일반 AR 국소 수정이다. diffusion의 우위는 "AR로 불가능"이 아니라 "정상 단계를 덜 망친다"는 좁은 마진일 수 있다. **그 마진이 없다면 "이 경우 diffusion은 불필요"가 결론이며, 그것도 유효한 결과로 기록한다.** 이 저울을 만드는 것이 A단계다.

---

## 관통 원칙: 검증이 스스로를 속이지 않게 하라

채점 도구는 "맞았다"고 말하기 쉽다. 아무것도 검사하지 않는 테스트도 통과하고, 검사기가 낸 답을 정답지로 삼으면 항상 일치한다. A단계의 거의 모든 설계 결정은 이 가짜 통과를 배제하는 데 맞춰졌다:

- **정답지를 검사기 출력으로 만들지 않는다.** 정답지는 시나리오 문서 기준으로 손으로 고정하고, 검사기 출력이 그와 일치하는지 대조한다.
- **정답지가 진짜 검증하는지 확인한다.** 경로 생성 로직을 일부러 망가뜨렸을 때 정답지 테스트가 실제로 실패하는지(mutation) 확인한다.
- **정밀도를 측정한다.** 놓친 오류(recall)뿐 아니라 지어낸 오류(precision)도 잰다 — 멀쩡한 곳을 짚으면 복구가 그곳을 망친다.
- **불일치를 숨기지 않는다.** 정적 검사와 실행 검사가 어긋날 때, 일치율을 억지로 100%로 만들지 않고 그 불일치가 무엇을 뜻하는지 기록한다.
- **범용성을 실험으로 보인다.** "하드코딩하지 않았다"를 주장이 아니라, 완전히 다른 도메인을 코드 수정 없이 통과시켜 증명한다.

---

## 채점의 세 축

계획을 세 축으로 채점한다:

- **구조** — 연결·순서·중복·사이클 등 계획이 형식적으로 올바른가.
- **의미(coverage)** — 계획이 task가 요구하는 근거·연산을 실제로 산출하는가. 각 단계가 "무엇을 만드는지"를 태그(`produces`)로 명시하고, 이름 규칙 없이 집합으로 대조한다(도메인 무관).
- **실행(runtime)** — 계획을 실제로 순회 실행하면 끝까지 도는가. 데이터는 흐르지 않고 호출의 성공/실패만 결정론적으로 판정한다.

세 축은 서로 다른 오류를 본다. 순서가 꼬였거나 종료 조건이 없는 계획은 구조 검사로는 걸리지만 실행은 성공한다("실행은 되지만 잘못 쓰인" 계획). 이 상보성이 세 축을 모두 두는 이유다.

---

## 구성

| 패키지 | 역할 |
| --- | --- |
| `plan_repair.schema` | `AgentTask` / `AgentPlan` / `Step`, corruption 메타데이터 |
| `plan_repair.canonical` | plan 정규화 — 결정론적 canonical JSON + 단계별 해시 |
| `plan_repair.validation` | 구조·의미 검사 + validator↔runtime 일치 판정 |
| `plan_repair.corruption` | corruption injector (단일 + 다중 오류 주입) |
| `plan_repair.runtime` | 결정론 실행 환경 |
| `plan_repair.data` | reference plan (도메인 A: 리서치 리포트 / 도메인 B: 데이터 분석) |
| `plan_repair.repair` | [B단계 예비] 복구기 port 자리 — 비어 있음 |

---

## 진행 상황

**A단계 — 평가 하네스 (완료)**

계획을 망가뜨리고, 그 오류를 정확히 탐지하는지 채점하는 시스템.

- **001** — 닫힌 루프: reference plan → 오류 주입 → validator → (주입 ↔ 탐지 대조). 기본 구조 검사.
- **002** — corruption 유형 확장 + 두 번째 도메인(리서치 리포트) 추가로 범용성 증명(코드 무수정 통과).
- **003** — 여러 오류를 동시에 주입, 탐지 정밀도(precision/recall)를 정량 측정. 겹친 오류를 정확히 분리해 짚는지 엄격 기준으로 검증.
- **004** — 의미 검사(coverage) + 결정론 실행 환경. 정적 검사와 실행 검사의 상보성 확인. **A단계 완료.**

**B단계 — 복구기 연결 (예정)**

- 복구기(결정론 / AR / diffusion)를 동일 인터페이스로 연결.
- validator가 좁힌 오류 구간을 diffusion에 넘겨 selective remask 복구.
- A단계의 세 축 채점으로 복구 품질을 측정하고 AR과 비교.

---

## 개발

```
uv sync            # 3.12 가상환경 + 의존성 (GPU 불필요)
uv run pytest      # 테스트 — 전부 모델 없이 돈다
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

핵심 의존성에 torch가 없다. 마스킹·토큰 정렬·채점·mock 백엔드는 전부 노트북에서 돌고,
diffusion 추론만 GPU가 필요하다.

## diffusion 실험 실행 (GPU)

```bash
git clone <repo> && cd diffusion_plan_repair
pip install -e '.[gpu]'          # torch + accelerate + bitsandbytes

python scripts/run_diffusion_experiment.py --model llada --out results/llada
python scripts/run_diffusion_experiment.py --model dream --out results/dream
```

케이스(모델 × 도메인 × corruption)마다 결과 JSON을 즉시 쓰고, 다시 실행하면 **없는 것만**
이어서 돈다 — 세션이 끊겨도 중간부터다. 진행바는 이미 끝난 케이스를 완료로 세고 시작한다.

```bash
--model {llada,dream,all}     --domain {domain_a,domain_b,all}
--corruption {broken_dependency,dependency_cycle,wrong_tool,duplicate_step,
              step_deletion,wrong_ordering,missing_stop_condition,drop_required_step,all}
--steps 64        # denoising 패스 수
--temperature 0   # 0이면 greedy(재현 가능), >0이면 샘플링
--limit N         # 새 케이스 N개만
--force           # 이미 끝난 것도 다시
```

GPU 없이 파이프라인만 확인하려면 백엔드를 mock으로 바꾼다:

```bash
python scripts/run_diffusion_experiment.py --backend oracle --out /tmp/dry
```
