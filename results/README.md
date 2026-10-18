# 공개 집계 결과

[summary.json](summary.json)은 실험 조건과 네 방식의 정수 집계값, [summary.csv](summary.csv)는 같은 값의 표 표현입니다.

출처는 기존 연구의 results/claim_b_20260915/analysis/sequence_run_codefix/sequence_summary.json에 저장된 aggregates 및 [제공된 논문](../paper/ICISC_2026_paper__52.pdf)의 Table 4(PDF 15쪽)입니다. 공개본의 JSON에는 원래 집계 파일의 SHA-256도 남겼습니다. 원래 집계 파일 전체를 공개한 것은 아닙니다.

- 단위: matches는 정답 토큰 위치, S/I/D/E는 토큰 편집 연산 개수입니다.
- 복원율 분모: 60×136=8,160개 정답 위치입니다.
- 구간 완전일치 분모: 60개 구간입니다.
- FC–L/FC–F: 같은 8개 view와 고정 좌표 처리에서 선택 규칙만 바꿉니다. 같은 최종 표본의 사후 탐색 분석입니다.
- CP–L/CP–F: 경계 입력도 달라지는 전체 방식 비교입니다.
- 원문·실제 복원문·token IDs·구간 식별자·임베딩·중간 상태를 이 표에 복사하지 않았습니다.
- 이번 공개 준비에서 원시 출력 재채점이나 구간별 재집계를 수행하지 않았습니다.

```bash
python3 tools/verify_summary.py
```

위 명령은 집계 산술과 파일 일치를 검산합니다. 조건과 한계는 [실험 안내](../docs/EXPERIMENTS.md)에 있습니다.
