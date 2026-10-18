# 실행 환경과 재현 범위

[처음으로](../README.md) · [구현 안내](IMPLEMENTATION.md)

## GPU·설치 없이 확인하는 범위

```bash
python3 tools/verify_summary.py
```

Python 표준 라이브러리만 사용합니다. 포함된 집계 JSON과 CSV의 일치, M+S+D=N·E=S+I+D·출력 길이, 대표 차이, 논문 PDF checksum과 원본 파일의 SHA-256을 확인합니다. 원본 연구 모듈을 import하거나 데이터·모델을 다운로드하지 않습니다.

이 명령은 **집계 산술·파일 일관성**을 검사합니다. 원시 출력 재채점과 모델 추론에는 별도의 자료와 실행환경이 필요합니다.

## 당시 연구 환경 기록

| 항목 | 보존된 조건 |
| --- | --- |
| OS / hardware | Linux aarch64 / NVIDIA GB10 |
| Python | 3.11.16 |
| GPU 환경 | CUDA 13.0, PyTorch 2.12.1+cu130 |
| 모델 라이브러리 | Vec2Text 0.0.13, Transformers 4.34.1, tokenizers 0.14.1 |
| 데이터·배열·설정 | datasets 2.14.6, NumPy 1.26.4, PyYAML 6.0.3 |

[pyproject.toml](../pyproject.toml)과 [requirements.txt](../requirements.txt)는 프로젝트의 의존성 선언을 그대로 보존했습니다. Python 범위는 >=3.10,<3.12이며 GPU 실험에는 위 Python 3.11 환경을 사용했습니다. 의존성 목록에는 초기 실험 단계의 패키지도 포함되어 있습니다.

다른 GPU·OS·라이브러리 버전에서 실행하려면 호환성 검사가 필요합니다.

## 기존 테스트 snapshot

공개한 기존 파일은 다음과 같습니다.

- [test_tokenizer_pairs.py](../tests/test_tokenizer_pairs.py): 합성 tokenizer의 원문 slice·EOS/PAD·중첩 geometry
- [test_chunking.py](../tests/test_chunking.py): 입력 구간 구성
- [test_diagnostics.py](../tests/test_diagnostics.py): 토큰 정렬·편집 오류·경계 처리
- [test_stage5d_assembly.py](../tests/test_stage5d_assembly.py): 고정 좌표 owner·삼중 관측
- [test_stage5c_fidelity_selector.py](../tests/test_stage5c_fidelity_selector.py): 선택 규칙과 평가용 gate
- [test_stage1_strong_saturation.py](../tests/test_stage1_strong_saturation.py), [test_stage5b_evaluate.py](../tests/test_stage5b_evaluate.py): 합성 평가·조건 처리
- [test_reconstruction.py](../tests/test_reconstruction.py), [test_data.py](../tests/test_data.py): adapter·입력·mock 데이터 호출. 일부 확인에는 설치된 PyTorch 등의 의존성이 필요합니다.

원시 연구 결과를 읽는 frozen-artifact 테스트는 공개 목록에서 제외했습니다. 포함된 테스트에는 PyYAML·NumPy·pytest·Faker 등의 의존성이 필요하며, 집계 검산 명령과 별도로 실행해야 합니다.

## 전체 추론·재채점이 바로 실행되지 않는 이유

공개본은 구현을 검토하고 집계표를 이해할 수 있는 snapshot입니다. 원시 corpus, 실제 prediction/reference token arrays·문자열, 후보 cache, 임베딩·모델 가중치·중간 checkpoint를 담지 않았습니다. 원본 설정이 참조하는 일부 freeze/cohort/cache 파일도 없습니다.

scripts의 실행에는 해당 단계의 frozen 입력과 모델·데이터 환경이 필요합니다. 필요한 파일이 없거나 manifest와 다르면 실행이 중단될 수 있습니다.

원고의 source-aligned selector와 oracle는 사후 진단입니다. 원문 미참조 결합 성능과 다른 범위이며 공개 집계의 토큰 복원율로 대체하지 않습니다.

## 공개 자료의 구성과 관리

- 집계 JSON/CSV는 논문과 저장된 실험 요약을 기준으로 정리했습니다.
- 코드·설정·테스트·PDF는 원본 그대로 보존하고 [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json)에 파일별 해시를 기록했습니다.
- 집계 검산은 모델 추론과 별도로 실행할 수 있도록 Python 표준 라이브러리로 구성했습니다.
