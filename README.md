# Verifier-Guided Diffusion Plan Repair

Agent가 생성한 실행 계획(plan)의 구조적 오류를 validator로 진단하고, diffusion LLM으로 오류 구간만 선택적으로 복구하는 연구용 실험 레포.

**원래 질문은 "diffusion이 정상 단계를 덜 망치면서(collateral 낮게) 복구하는가"였다. 측정된 답은 아니오 — 무승부다.** diffusion이 푸는 유형에서 ar_local과 collateral을 나란히 재면 둘 다 0이다. 그러나 그 무승부를 파고들며 더 일반적인 것이 나왔다: **복구 정밀도(collateral)를 정하는 변수는 복구기의 종류가 아니라 "오류 영역을 얼마나 좁게 지목했는가"다.** 같은 모델·같은 손상에서 영역만 좁히면 collateral이 10→0으로 떨어지고(diffusion, 인과), 영역을 지목받은 자유 생성 복구기는 그 영역 안에 머물며(ar_local), 영역을 지목받지 못한 같은 모델은 상한을 넘긴다(ar_full, 음성 대조).

> A(채점)·B(복구기 연결)·C(실패 원인 규명) 완료. **D단계 진행 중** — diffusion을 `wrong_tool`·`broken_dependency`에서 작동시켰고(스냅·마스크 정밀화), 방향 1에서 "복구 정밀도는 복구기가 아니라 영역 지정이 정한다"를 확인했다. `dependency_cycle` solved, ar 반복 측정, 커버리지 확장(사거리 밖 5유형)은 향후 과제. 아래 "진행 상황" 참고.

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
| `plan_repair.repair` | 복구기 5종 + selective remask + 복구 채점(4갈래 collateral) |

---

## 진행 상황

**A단계 — 평가 하네스 (완료)**

계획을 망가뜨리고, 그 오류를 정확히 탐지하는지 채점하는 시스템.

- **001** — 닫힌 루프: reference plan → 오류 주입 → validator → (주입 ↔ 탐지 대조). 기본 구조 검사.
- **002** — corruption 유형 확장 + 두 번째 도메인(리서치 리포트) 추가로 범용성 증명(코드 무수정 통과).
- **003** — 여러 오류를 동시에 주입, 탐지 정밀도(precision/recall)를 정량 측정. 겹친 오류를 정확히 분리해 짚는지 엄격 기준으로 검증.
- **004** — 의미 검사(coverage) + 결정론 실행 환경. 정적 검사와 실행 검사의 상보성 확인. **A단계 완료.**

**B단계 — 복구기 연결 (구현 완료, 실측 1회차)**

계획을 실제로 고치고, A단계 저울로 채점한다.

- **B-1** — 복구기 Protocol(모두 같은 시그니처) + 결정론 baseline + mock(하한/상한) + 복구 채점.
- **B-2** — AR 복구기 2모드(전체 재생성 / 국소 수정). GPT-5 API, 파싱 실패 = 복구 실패.
- **B-2b** — collateral을 4갈래(modified/renamed/removed/added)로 분리. 저울의 사각지대 제거.
- **B-3a** — selective remask 로직: validator가 짚은 step만 마스크, 나머지는 구조적으로 보존.
- **B-3b-1** — 문자 span → 토큰 index 정렬(LLaDA/Dream 실제 tokenizer로 검증).
- **B-3b-2** — PyTorch 백엔드로 실제 denoising + 재개 가능한 실험 러너.

### 실측 결과 (B단계, `results/`, 셀당 1회)

복구기 5종 × corruption 8유형 × 도메인 2개 = 80 셀. 아래는 **B단계 시점**의 수이고, D단계·방향 1에서
집계기를 수리한 뒤 일부 값이 바뀌었다(아래 각주).

| 복구기 | 측정된 셀 | solved | collateral 합 |
| --- | --- | --- | --- |
| deterministic | 16/16 | 6 | 0 |
| ar_full (전체 재생성) | 16/16 | 2 | 30 |
| ar_local (국소 수정) | 16/16 | **9** | 2 |
| LLaDA (diffusion) | 11/16 | **0** | 18 |
| Dream (diffusion) | 4/16 | **0** | 0 |

> **집계기 수리 후 정정.** 옛 집계기는 (1) 같은 셀의 여러 측정 중 디렉터리 사전순 마지막만 남기고
> (2) api 빈 응답을 "측정됨(collateral 0)"으로 셌다. 수리 후: **ar_full은 16/16이 아니라 2/16 측정**
> (14셀이 gpt-5 빈 응답, 실제 측정 2셀의 collateral 평균은 15), **ar_local은 4셀이 빈 응답**,
> 최신 측정이 여럿인 셀은 승자를 지어내지 않고 "측정 여럿"으로 표시한다. 자세한 것은 D단계·방향 1.

**밝혀진 것:**

- **diffusion은 이 설정에서 하나도 풀지 못했다(0/15).** LLaDA는 유효한 계획을 만들었지만(10/11 파싱 성공)
  `produces` 태그를 재생성하면서 원본과 다른 이름을 지어내 의미 검사에 걸렸다(`missing_operation` 8셀).
  step **전체**를 마스크한 결과이지 모델의 무능이 아니다 — 필드 단위 마스킹이 다음 과제다.
- **가장 잘 푼 것은 ar_local이다(9/16), collateral 2로.** 계획 문서가 "diffusion의 진짜 경쟁자"로 지목한 방식이다.
- **전체 재생성의 대가가 수치로 나왔다** — ar_full 30 vs ar_local 2. 다만 이 비교는 AR 두 모드 사이의 것이고,
  diffusion은 아직 여기에 참여할 수준이 아니다.
- **결정론 baseline이 6/16을 푼다.** 정보가 사라지지 않은 오류(순서·중복·종료조건)는 전부, 사라진 오류는 하나도.
- **15셀 중 11셀이 OOM으로 측정조차 되지 않았다**(Dream 9셀). Dream은 현재 데이터로 평가할 수 없다.

정직하게: 셀당 1회 실행이고, GPT-5는 temperature를 못 낮춰 비결정적이다. 자세한 것은
[docs/design-decisions-phase-b.md](docs/design-decisions-phase-b.md).

**C단계 — 실패 원인 규명 (초안)**

B단계는 diffusion 0/15가 모델의 한계인지 우리 파이프라인의 한계인지 갈리지 않은 채 끝났다.
C단계는 원인을 하나씩 제거하고 그때마다 다시 쟀다. **앞의 두 가설은 틀렸고, 그 오답이 남긴
데이터가 진짜 원인을 가리켰다.**

- **C-1 필드 마스킹** — 깨진 필드(`tool`/`input_from`)만 마스크해 `produces`를 보존.
  마스크가 87~92% 좁아지고 `missing_operation`이 사라졌다. **그러나 여전히 실패**(`unknown_tool`).
  가설은 표면 원인만 맞혔다.
- **C-2 유효 tool 힌트** — "모델이 유효 어휘를 모른다" 가설. ar_local만 받던 tool 목록을
  diffusion에도 프리픽스로 줘 **비교의 불공정을 제거**했다. **가설은 틀렸다** — 목록을 줘도 실패
  (`dedupe`→`deduplicate`, `join`→`join_db`).
- **토큰 분석** — "정답을 알면서 복사하지 않는다"는 가설도 틀렸다. 모델은 `join`을 **정확히 썼고**,
  마스크가 정답보다 1토큰 길어 남는 칸에 `_db`를 붙인 것이었다. 원인은 손상 규약 `_x`가
  정확히 1토큰이라는 데 있었다(39/39 스텝에서 잉여 1).
- **C-3 길이 맞춘 손상** — 정답과 토큰 길이가 같은 손상으로 대조군을 만들어 마스크 = 정답 길이로
  통제. **결과: solved.**

| 단계 | 마스크 칸 | 모델이 채운 것 | 결과 |
| --- | --- | --- | --- |
| C-1 · A/B | 5 / 4 | `"deduplicate"` / `"merge_join"` | `unknown_tool` |
| C-2 · A/B | 5 / 4 | `"deduplicate"` / `"join_db"` | `unknown_tool` |
| C-3 · A/B | **4 / 3** | **`"dedupe"` / `"join"`** | **solved, collateral 0, 복원 1/1** |

**출력 토큰 수는 여섯 케이스 전부에서 마스크 칸 수와 정확히 일치한다.** 잉여 칸이 있으면
반드시 채워진다.

C-2와 C-3는 복구기 설정이 완전히 동일하고, 모델이 받는 토큰열도 **마스크 칸을 빼면 완전히 같다**
(906 vs 905, 890 vs 889토큰). 따라서 두 결과의 차이는 마스크 길이에 귀속된다.

**정직한 스코프.** 이것은 "`wrong_tool` 유형에서, 마스크 길이를 정답과 맞췄을 때, LLaDA가 정확히
복구하며 정상 단계를 하나도 건드리지 않는다"는 2케이스 결과다. 모든 유형·모든 조건이 아니다.
여전히 열려 있는 것: `broken_dependency`(다른 실패 양상), Dream(미재측정), 힌트의 필요성(길이만
맞춘 조건은 미측정), 일반 `_x` 손상에서의 마스크 초과(원인 규명일 뿐 해결 아님), 같은 조건에서의
AR 비교(프로젝트의 원래 질문은 아직 답해지지 않았다), OOM 11셀.

자세한 것은 [docs/design-decisions-phase-c.md](docs/design-decisions-phase-c.md).

**D단계 — 목표 전환: 측정에서 작동으로 (진행 중)**

C단계까지는 **복구 로직을 건드리지 않는 것**이 원칙이었다. 질문이 "diffusion이 무엇을 할 수
있는가"였고, 답을 몰래 고쳐 주는 파이프라인은 파이프라인을 재기 때문이다. D단계의 목표는
"정확히 작동하는 복구 시스템"이며, **이 단계의 변경은 측정이 아니라 개선이다.** 그렇게 읽어야
하고, 그렇게 기록한다.

- **D-1 유효 tool 스냅** — 모델이 채운 tool이 유효 tool을 **앞에서부터 거의 전부 재현**했을 때만
  (문자 최장공통접두사 비율 ≥ 0.8, 유일) 그 유효 tool로 완성한다. `join_db` → `join`,
  `deduplicate` → `dedupe`. **`merge_join`은 스냅하지 않는다** — 값 안에 숨은 이름을 찾는 것은
  완성이 아니라 검색이고, 그 허용이 스냅을 도장 찍기로 만든다.
  denoise 이후의 후처리라 복구 로직은 그대로다(`dllm_backend_torch.py` 무변경).
- **BD-1 파생 dangling 제외** — 참조 하나가 깨지면 validator는 둘을 보고한다: 깨진 참조와,
  **그 때문에 소비자를 잃은 건강한 스텝**. 후자는 손상이 아니라 결과인데 마스크가 그것을 통째로
  덮어(마스크의 72~81%) 모델이 다시 쓰게 만들었다. 이제 **깨진 참조로 설명되는 dangling만**
  마스크에서 뺀다 — 중복이 남긴 dangling처럼 스스로 매달린 스텝은 그대로 마스크한다.
  판정은 validator가 아니라 마스크 쪽에서 한다(validator는 측정 도구이므로 불변).
- **BD-2 input_from 후처리** — 좁힌 마스크에서도 채움이 두 갈래로 틀렸다: 도메인 B는 정답 id를
  정확히 쓰고 잉여 칸을 **빈 원소**로 채웠고, 도메인 A는 id 자리에 **`produces` 태그**를 썼다.
  빈 원소는 버리고, 태그는 **그것을 내는 스텝이 하나일 때만** 그 id로 바꾼다. 도메인 B에는
  두 스텝이 함께 내는 태그가 셋 있어(`normalization` 등) **실제로 거부가 발생하며**, 거부도
  기록한다 — 보수성의 비용을 셀 수 있어야 하기 때문이다.

문턱 0.8은 네 관측에 맞춘 값이 아니다. **두 도메인의 유효 tool끼리 이 비율은 최대 0.71**이라
어떤 유효 tool도 다른 유효 tool로 스냅될 수 없다 — 어휘의 구조적 성질이고 테스트로 고정돼 있다.

기본은 **off**다(`--snap`로 켠다). 켜면 아무것도 하지 않는 대조군(`--backend echo`)이 손상의
`_x` 접미사째로 구제돼 solved가 되므로, 대조군은 스냅 없이 돌아야 한다.

**실측(GPU).** 스냅을 켜고 다시 재니 `wrong_tool` 2셀이 solved, collateral 0이 됐다. 모델은
여전히 `deduplicate`·`join_db`를 냈고(원본 출력 보존), 스냅이 그것을 유효 tool로 완성해 solved가
됐다 — "모델이 맞췄나 스냅이 찍었나"를 가르도록 원본과 스냅 기록을 분리해 남긴다. `broken_dependency`도
BD-1(마스크 정밀화)로 collateral이 1→0이 되고, BD-2(input_from 후처리)로 두 도메인 다 solved가 됐다.

| 유형 | 측정 | 마스크 토큰 | solved | collateral |
| --- | --- | --- | --- | --- |
| `wrong_tool` A/B | B단계 (스텝 전체) | 46 / 37 | ✗ | 0 |
| `wrong_tool` A/B | D-1 (필드+스냅) | **5 / 4** | **✓** | **0** |
| `broken_dependency` B | B단계 (2스텝) | 76 | ✗ | **1** |
| `broken_dependency` B | BD-1 (영역 축소) | **9** | ✗ | **0** |
| `broken_dependency` A/B | BD-2 (+후처리) | 15 / 9 | **✓** | **0** |

**D단계 재측정 (`--snap`/`--snap-deps` 켜고, 대조군은 끄고):**

| 파일 | 내용 |
| --- | --- |
| `results/diffusion_d1/` | `wrong_tool` + 스냅, 2셀 solved |
| `results/diffusion_bd1/` | `broken_dependency` 마스크 정밀화, collateral 1→0 |
| `results/diffusion_bd2/` | `broken_dependency` + 후처리, 2셀 solved |

---

**방향 A — collateral 비교: 원래 질문에 답하다 (무승부)**

diffusion이 푸는 유형(`wrong_tool`, `broken_dependency`)에서 diffusion과 ar_local의 collateral을
나란히 쟀다. **양쪽 다 0 — 무승부.** 원래 가설이 예상한 "diffusion이 더 깔끔하다"의 마진은
관측되지 않았다.

그리고 그 무승부가 실마리를 줬다. BD-1이 **마스크를 좁히자** collateral이 1→0이 된 것 —
즉 **diffusion의 collateral 0은 모델의 성질이 아니라 마스크 범위(영역 지정)의 성질**이었다.

**방향 1 — 복구 정밀도는 복구기가 아니라 영역이 정한다**

그렇다면 진짜 독립변수는 "영역 정밀도"다. 유형을 고정하고 영역만 바꿔 검증했다.

- **diffusion within-type 인과 (2건).** `dependency_cycle` A에서 같은 11스텝(정상 10개 동일)을
  마스크하되 스텝 전체(387토큰)가 아니라 `input_from` 필드만(51토큰) 열었다. **collateral 10→0.**
  정상 스텝 수는 그대로고 바뀐 것은 스텝 *안*의 노출 범위뿐이다. B단계에서는 그 10스텝을 지우고
  10개를 새로 썼는데, 좁힌 영역에서는 건드릴 문자가 `input_from` 값뿐이라 하나도 못 건드린다.
  `broken_dependency` B(1→0)도 같은 성격. **못 풀어도(사이클은 여전히 안 끊긴다) 영역을 좁히면
  안 망친다** — collateral은 solved와 독립이고 영역이 정한다.
- **ar도 영역 안에 머문다.** ar_local은 계획 전체를 다시 쓰는데도(마스크 같은 강제 없음)
  `dependency_cycle`(영역 11·15스텝, 예산 상향 후 solved)에서 **손상 1개만 고치고 정상 18·19스텝을
  보존**했다. collateral 0.
- **음성 대조 — 같은 모델, 영역을 안 주면 상한을 넘는다.** ar_full은 validator 결과를 프롬프트에
  쓰지 않는다("처음부터 새로 써라"). 같은 셀·같은 gpt-5인데 `dependency_cycle` A에서 **collateral 15**
  (정상 18스텝 중 15개를 흔듦). **모델이 아니라 영역 지목이 정한다는 핵심 근거.**
- **상한.** "collateral ≤ 영역 내 정상 스텝 수"가 63건 중 **61건**에서 지켜졌고, 위반 2건은
  **둘 다 ar_full**(영역을 안 듣는 복구기)이다. diffusion에서는 코드로 강제되고(마스크 밖 쓰기 거부),
  ar_local에서는 지시와 지목만으로 지켜진다 — 기전이 다른데 결과가 같다.

**함의.** collateral을 낮추려면 더 정교한 복구기를 고르는 것보다, validator가 오류 영역을
정밀하게(파생 증상과 근본 원인을 구분해) 지목하게 하는 것이 지름길이다. BD-1(파생 dangling 제외)이
그 실례다 — 마스크를 좁히자 collateral이 사라졌다.

**정직한 스코프.**
- **커버리지는 ar 우위.** diffusion은 8유형 중 3유형만 사거리 안이다(`wrong_tool`,
  `broken_dependency`, collateral만 0인 `dependency_cycle`). 나머지 5유형(삽입/삭제/이동/계획 필드)은
  마스크 표현력 밖이며, 이를 넘는 방법(FlexMDM 등 가변 길이 diffusion)은 LLaDA 파인튜닝(≈1000 H100h)이
  필요해 자원 밖이다. training-free 접근(ρ-EOS)은 LLaDA가 마스크 안에서 eos를 내지 않아 막혔다.
- **ar의 큰 영역 결측은 능력이 아니라 예산이었다.** `max_completion_tokens=4096`이 추론에 소진돼
  빈 응답이 됐고(`finish_reason=length`), 16384에서 solved. 이걸 뚫지 않았다면 "ar은 큰 영역을 못 한다"는
  가짜 결론을 썼을 것이다.
- **diffusion의 solved는 후처리에 의존한다.** ar_local은 후처리 없이 solved다. 무승부 표는 이
  비대칭과 함께 읽어야 한다.
- **within-type 인과는 2건, 셀당 1회다.** between-type 상관(61/63)이 보강하나 교락을 끊은 증거는 2건이고,
  ar은 비결정인데 반복 측정하지 않았다. LLaDA만, 도메인 2개.

자세한 것은 `docs/design-decisions-direction-1.md`(로컬 자산).

---

## 실측 재현

```
uv run python scripts/run_diffusion_experiment.py --model deterministic --out results/deterministic
uv run python scripts/run_diffusion_experiment.py --model ar_local --out results/ar   # OPENAI_API_KEY 필요
uv run python scripts/aggregate_results.py --results results                          # 표 전부 출력
```

`results/`의 각 JSON은 한 셀이다 — 채점 결과, 복구된 계획, 그리고 diffusion의 경우 모델 raw 출력과
파싱 실패 진단(마스크 토큰 잔존 수, 깨진 지점 문맥)까지 담는다. 집계 표의 모든 숫자는
`scripts/aggregate_results.py`가 이 파일들에서 계산하므로, 다시 돌려 확인할 수 있다.

C단계 재측정은 GPU에서 돌린 것을 단계별로 따로 뒀다:

| 파일 | 내용 |
| --- | --- |
| `results/diffusion_c1/c1_remeasure.json` | 필드 마스킹 (힌트 없음), 4케이스 |
| `results/diffusion_c2/c2_remeasure.json` | + 유효 tool 힌트, `wrong_tool` 2케이스 |
| `results/diffusion_c3/c3_remeasure.json` | + 길이 맞춘 손상, 2케이스 (solved) |

D단계·방향 1 재측정:

| 파일 | 내용 |
| --- | --- |
| `results/diffusion_d1/` | `wrong_tool` + 스냅, 2셀 solved |
| `results/diffusion_bd1/` · `bd2/` | `broken_dependency` 마스크 정밀화 → 후처리, collateral 1→0 → solved |
| `results/diffusion_wt/within_type_remeasure.json` | `dependency_cycle` 영역 축소, collateral 10→0 (within-type 인과) |
| `results/ar_diag/` · `ar_big/` | ar `dependency_cycle` 예산 4096(빈 응답) vs 16384(solved, coll 0) |

각 파일의 `masked_token_count`, `diagnostics.fillings`, `diagnostics.snaps`/`dependency_snaps`,
`backend_diagnostics.hint_tokens`, `injected[].detail`이 위 표의 근거다. C-3의 `injected[].detail`은 손상 모드와 길이 매칭에 쓴
tokenizer까지 기록하므로, 어느 손상으로 얻은 수치인지 파일만 보고 구별된다.

길이 맞춘 손상으로 다시 돌리려면:

```
python scripts/run_diffusion_experiment.py --model llada \
    --corruption wrong_tool_length_matched --out results/diffusion_c3
```

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
              step_deletion,wrong_ordering,missing_stop_condition,drop_required_step,
              wrong_tool_length_matched,all}   # all은 앞의 8개(기존 매트릭스 그대로)
--match-tokenizer llada       # 길이 맞춘 손상을 어느 어휘로 매칭할지
--snap            # 채운 tool 이름을 유효 tool로 완성(D-1). 기본 off — 대조군은 끄고 돈다
--snap-deps       # 채운 input_from에서 빈 원소 제거 + 태그→step id(유일할 때만) (BD-2). 기본 off
--steps 64        # denoising 패스 수
--temperature 0   # 0이면 greedy(재현 가능), >0이면 샘플링
--limit N         # 새 케이스 N개만
--force           # 이미 끝난 것도 다시
```

GPU 없이 파이프라인만 확인하려면 백엔드를 mock으로 바꾼다:

```bash
python scripts/run_diffusion_experiment.py --backend oracle --out /tmp/dry
```