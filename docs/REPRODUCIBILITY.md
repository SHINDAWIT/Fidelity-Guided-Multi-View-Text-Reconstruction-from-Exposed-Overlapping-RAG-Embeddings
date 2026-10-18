# 실행 환경과 재현 범위

[처음으로](../README.md) · [구현 안내](IMPLEMENTATION.md)

## GPU·설치 없이 확인하는 범위

```bash
python3 tools/verify_summary.py
```

Python 표준 라이브러리만 사용합니다. 포함된 집계 JSON과 CSV의 일치, M+S+D=N·E=S+I+D·출력 길이, 대표 차이, 제공된 PDF checksum, 복사한 원본 파일의 SHA-256을 확인합니다. 원본 연구 모듈을 import하거나 데이터·모델을 다운로드하지 않습니다.

이 검사는 **공개 패키지의 산술·파일 일관성 검사**입니다. 논문 실험의 재실행, 원시 출력 재채점, 모델 동작 검증, 독립적인 결과 재현을 의미하지 않습니다.

## 당시 연구 환경 기록

| 항목 | 보존된 조건 |
| --- | --- |
| OS / hardware | Linux aarch64 / NVIDIA GB10 |
| Python | 3.11.16 |
| GPU 환경 | CUDA 13.0, PyTorch 2.12.1+cu130 |
| 모델 라이브러리 | Vec2Text 0.0.13, Transformers 4.34.1, tokenizers 0.14.1 |
| 데이터·배열·설정 | datasets 2.14.6, NumPy 1.26.4, PyYAML 6.0.3 |

[pyproject.toml](../pyproject.toml)과 [requirements.txt](../requirements.txt)는 당시 프로젝트의 의존성 선언을 그대로 보존했습니다. pyproject의 Python 범위는 >=3.10,<3.12이며 GPU 실험은 위 Python 3.11 환경으로 기록되어 있습니다. 이 선언에 여러 초기 단계 의존성이 포함되어 있다는 사실이 모두를 최종 논문에서 사용했다는 뜻은 아닙니다. 예를 들어 openai 패키지의 존재를 유료 API 사용 경험으로 해석하지 않습니다.

이 환경을 이번 공개 준비에서 재설치하거나 모든 플랫폼에서 설치 가능함을 검증하지 않았습니다.

## 기존 테스트 snapshot

공개한 기존 파일은 다음과 같습니다.

- [test_tokenizer_pairs.py](../tests/test_tokenizer_pairs.py): 합성 tokenizer의 원문 slice·EOS/PAD·중첩 geometry
- [test_chunking.py](../tests/test_chunking.py): 입력 구간 구성
- [test_diagnostics.py](../tests/test_diagnostics.py): 토큰 정렬·편집 오류·경계 처리
- [test_stage5d_assembly.py](../tests/test_stage5d_assembly.py): 고정 좌표 owner·삼중 관측
- [test_stage5c_fidelity_selector.py](../tests/test_stage5c_fidelity_selector.py): 선택 규칙과 평가용 gate
- [test_stage1_strong_saturation.py](../tests/test_stage1_strong_saturation.py), [test_stage5b_evaluate.py](../tests/test_stage5b_evaluate.py): 합성 평가·조건 처리
- [test_reconstruction.py](../tests/test_reconstruction.py), [test_data.py](../tests/test_data.py): adapter·입력·mock 데이터 호출. 일부 확인에는 설치된 PyTorch 등의 의존성이 필요합니다.

원시 연구 결과를 직접 읽는 frozen-artifact 감사 테스트는 공개 기본 목록에서 제외했습니다. 포함된 테스트도 PyYAML·NumPy·pytest·Faker 등 설치 의존성이 있어 Python 표준 라이브러리만으로 실행되는 것은 아닙니다.

**이번 공개 준비에서 기존 pytest suite나 연구 코드는 실행하지 않았습니다.** 과거 서로 다른 테스트 로그의 통과 수를 합산하거나 이번 통과 수로 표시하지 않습니다.

## 전체 추론·재채점이 바로 실행되지 않는 이유

공개본은 구현을 검토하고 집계표를 이해할 수 있는 snapshot입니다. 원시 corpus, 실제 prediction/reference token arrays·문자열, 후보 cache, 임베딩·모델 가중치·중간 checkpoint를 담지 않았습니다. 원본 설정이 참조하는 일부 freeze/cohort/cache 파일도 없습니다.

scripts의 원본 진입점은 검토용으로 보존했으며, 이를 곧바로 실행하면 필요한 frozen 입력이나 환경이 없어 거부될 수 있습니다. manifest 검사를 우회하거나 입력 파일을 가짜로 만들어 논문 재현을 주장하지 않습니다.

원고의 source-aligned selector와 oracle는 사후 진단입니다. 원문 미참조 결합 성능과 다른 범위이며 공개 집계의 토큰 복원율로 대체하지 않습니다.

## 공개본 생성 시 확인한 범위

- 제공된 PDF의 제목·핵심 조건·대표 수치와 기존 기록을 대조했습니다.
- 원본에서 복사한 코드·설정·테스트·PDF를 바꾸지 않았습니다. [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json)은 공개 snapshot의 파일 동일성을 기록합니다.
- 집계 JSON/CSV를 기존 요약의 숫자로 구성하고 산술·상대 링크·소스 import 관계를 정적으로 검토했습니다.
- 신규 실험, 모델 추론, 연구 코드·테스트 실행, 패키지 설치, 모델·데이터 다운로드는 수행하지 않았습니다.
- 새 GitHub commit은 공개 snapshot 준비를 기록하며 과거 연구 기간·개인별 저작 이력·독립적인 사전등록을 입증하지 않습니다.
