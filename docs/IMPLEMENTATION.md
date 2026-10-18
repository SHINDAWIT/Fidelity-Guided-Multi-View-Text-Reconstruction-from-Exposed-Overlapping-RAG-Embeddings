# 구현을 읽는 순서

[처음으로](../README.md) · [논문](../paper/ICISC_2026_paper__52.pdf) · [실험·결과](EXPERIMENTS.md)

연구 코드의 입력 검증, 기존 모델 연동, 출력 선택·결합과 평가 흐름을 설명합니다. src·scripts·configs·tests는 원본 그대로 보존했으며, 파일별 원본 상대경로와 SHA-256은 [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json)에 기록했습니다.

## 1. 입력 표현과 기존 모델의 계약

**문제:** 전체 문서의 토큰을 잘라 얻은 ID와 그 범위의 문자열을 독립적으로 토큰화한 ID가 다를 수 있습니다. 이 차이가 남으면 입력 표현 변화가 복원·결합 오류와 섞입니다.

- [chunking.py](../src/overlap_poc/chunking.py#L226)의 find_roundtrip_stable_chains는 전체 토큰 offsets에서 문자열을 추출하고, 특수 토큰 없이 독립 토큰화한 ID가 의도한 slice와 같은지 검사합니다.
- _prepare_standalone_window는 내용 토큰·특수 토큰, 길이, padding, attention mask를 확인합니다. 주 설정은 **31개 내용 토큰 + EOS = 32개 모델 위치**입니다.
- [reconstruction.py](../src/overlap_poc/reconstruction.py#L229)의 validate_inputs·embed_raw_windows는 모델 호출 전 길이 및 expected input IDs를 확인합니다.

이 처리는 조건에 맞는 입력을 선별하거나 거부합니다. 불안정 입력을 모두 정상 복원하는 처리가 아닙니다. [test_tokenizer_pairs.py](../tests/test_tokenizer_pairs.py)는 합성 tokenizer로 EOS/PAD·중첩 구조를 다루는 기존 테스트입니다.

## 2. 기존 Vec2Text와 프로젝트 추가 처리

[Vec2TextBackend](../src/overlap_poc/reconstruction.py#L175)는 공개 pretrained inverter/corrector를 호출하는 adapter입니다. 후보 생성·교정·sequence-beam 검색·cosine 순위는 기존 라이브러리의 기능입니다.

generate_final_sbeam_candidate_pools는 기존 최종 후보를 관찰하고 공개 Top-1과 token-ID 일치를 요구하는 계약을 둡니다. Candidate에는 복원문, content IDs, local_score와 추론 조건이 저장됩니다. 실제 후보·복원문·배열 파일은 이 공개본에 포함하지 않았습니다.

**기존 점수:** 복원문을 다시 임베딩한 값과 해당 목표 임베딩 사이의 cosine 일치도입니다. 하나의 복원문 전체에 대한 점수이며 토큰별 정답 확률이나 새로 학습한 신뢰도 점수가 아닙니다.

## 3. 고정 좌표 선택·결합: FC–L / FC–F

[fixed_coordinate_fidelity_assembly](../src/overlap_poc/stage5d_assembly.py#L163)는 후보와 알려진 geometry만 받습니다.

- 시작 위치는 view index×15, 명목상 내용 길이는 31입니다.
- 정확히 두 view가 덮는 위치에서는 local_score가 높은 view를 선택합니다.
- 1e-12 이내 동률, 단일 관측, 삼중 관측은 낮은 view index를 선택합니다.
- 선택한 출력이 짧아 해당 위치가 비어 있으면 생략합니다. 다른 view로 메우지 않습니다. 명목상 31 위치 뒤의 토큰은 제외합니다.
- FC–L은 같은 함수의 점수를 모두 공통값으로 둬 앞 view 우선을 유도한 사후 replay 대조군입니다.

이 때문에 FC–L 대 FC–F는 같은 입력·좌표·예외 처리를 유지한 선택 규칙 비교가 됩니다. [기존 합성 테스트](../tests/test_stage5d_assembly.py)는 multiplicity-two 선택과 triple endpoint를 구분합니다.

## 4. 경계 어긋남을 고려한 정렬형 결합: CP–F

[assemble_source_free_progressive](../src/overlap_poc/stage6a_source_free_assembly.py#L235)는 누적 출력 끝부분과 다음 view 앞부분을 정렬합니다. [diagnostics.py](../src/overlap_poc/diagnostics.py)의 constrained stitcher를 사용합니다.

경계 폭은 명목상 overlap 16에서 ±2인 **14–18** 범위이고, 목적함수는 다음과 같습니다.

```text
편집 거리 + 0.5 × (|왼쪽 폭 − 16| + |오른쪽 폭 − 16|)
```

폭과 비용은 개발 시 정한 heuristic입니다. 정렬 후 owner의 토큰을 사용하고 owner 쪽이 비어 있으면 해당 열을 생략합니다. 내부 삼중 관측 endpoint에는 앞쪽 토큰 유지 규칙이 있습니다.

누적 출력의 왼쪽 점수에는 **직전 원래 view의 점수**를 대리값으로 사용합니다. 토큰별 provenance 점수를 추적하는 구조가 아닙니다.

FC–F와 CP–F는 경계 추정·길이 처리도 달라집니다. CP–L 역시 원래 인접 후보쌍을 경계 입력으로 사용하므로, CP–L 대 CP–F를 선택 규칙만의 효과로 해석하지 않습니다.

## 5. 결합과 정답 기반 평가의 분리

[고정 좌표 결합 함수](../src/overlap_poc/stage5d_assembly.py#L163)와 progressive 함수에는 정답 원문을 넘기지 않습니다. 이후 [_document_score](../src/overlap_poc/stage5d_assembly.py#L200)는 최종 출력과 정답 reference를 정렬해 지표를 구합니다.

- 토큰 복원율: 정확히 맞춘 **정답 위치 수 / 정답 위치 수**
- 삽입: 예측에만 있는 추가 토큰
- 삭제: 놓친 정답 토큰
- 치환: 대응 위치에서 잘못 복원한 토큰
- 구간 완전일치: 예측 전체가 선택된 136-token 정답 구간과 같은지

과거 필드 이름 document_exact가 있어도 평가 대상은 문서 전체가 아닌 선택된 구간입니다. [stage5c_fidelity_selector.py](../src/overlap_poc/stage5c_fidelity_selector.py)의 원문 대응 진단과 oracle는 평가용이며 실제 원문 미참조 출력과 구분합니다.

## 6. 저장과 재개 조건

[stage6c_strong.py](../src/overlap_poc/stage6c_strong.py#L149)의 _fingerprint는 알고리즘·코호트·입력 manifest·임베딩·모델 설정·추론 조건을 연결합니다.

_load_checkpoint는 식별값, 후보·record·완료 key의 길이, 예정 순서의 prefix와 완료 목록이 같은지를 확인하고 불일치하면 재개를 거부합니다.

## 주요 실행 진입점

| 원본 진입점 | 역할 | 공개본의 범위 |
| --- | --- | --- |
| [run_stage3a_longdoc_feasibility.py](../scripts/run_stage3a_longdoc_feasibility.py) | 토큰화 안정 입력 구성 | 실행 코드·관련 설정만 포함 |
| [freeze_stage6b_algorithm.py](../scripts/freeze_stage6b_algorithm.py) | 알고리즘 기록 동결 | 연결되는 기존 연구 artifact는 미포함 |
| [freeze_stage6c_final_cohort.py](../scripts/freeze_stage6c_final_cohort.py) | 최종 코호트 구성 | 원문·코호트 raw 파일은 미포함 |
| [run_stage6c_strong.py](../scripts/run_stage6c_strong.py) | 기존 모델 추론·캐시·중간 저장 | 모델·임베딩·후보 cache는 미포함 |
| [run_stage6c_final_evaluation.py](../scripts/run_stage6c_final_evaluation.py) | 출력 결합과 사후 평가 | frozen 출력·정답 배열은 미포함 |

실행 가능 범위는 [재현 안내](REPRODUCIBILITY.md)를 먼저 확인할 수 있습니다.
