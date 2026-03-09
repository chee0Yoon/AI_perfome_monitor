# Processing prod_data_sample.csv with final_metric_refactor

이 가이드는 `prod_data_sample.csv`를 전처리하고 `final_metric_refactor`을 실행하는 방법을 설명합니다.

## 개요

`prod_data_sample.csv`의 데이터는 복잡한 JSON 형식의 input/output을 가지고 있습니다:
- **input**: 사용자 프롬프트와 시스템 메시지를 포함한 메시지 배열
- **output**: 모델의 응답을 포함한 JSON

이 스크립트는:
1. **input**에서 **user_context** (사용자 메시지)를 추출
2. **output**에서 **assistant_response** (모델 응답)를 추출
3. 평가 점수(correctness 등)를 추출
4. 최소한의 필요한 형식으로 새 CSV를 생성
5. final_metric_refactor을 자동으로 실행

## 사용 방법

### 1. 전처리만 수행

```bash
python3 scripts/preprocess_and_run_prod_data.py --preprocess-only
```

결과: `data/prod_data_processed_YYYYMMDD_HHMMSS.csv` 생성

### 2. 전처리 + final_metric_refactor 실행 (통합 모드)

```bash
python3 scripts/preprocess_and_run_prod_data.py
```

또는:

```bash
python3 scripts/preprocess_and_run_prod_data.py \
  --source data/prod_data_sample.csv \
  --mode integrated \
  --max-rows 0
```

### 3. 제한된 행 수로 실행 (상세 모드)

```bash
python3 scripts/preprocess_and_run_prod_data.py \
  --source data/prod_data_sample.csv \
  --max-rows 100 \
  --mode detailed
```

### 4. 커스텀 출력 디렉토리 지정

```bash
python3 scripts/preprocess_and_run_prod_data.py \
  --source data/prod_data_sample.csv \
  --output-dir /path/to/output
```

## 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source` | 입력 CSV 경로 | `data/prod_data_sample.csv` |
| `--max-rows` | 처리할 최대 행 수 (0=전체) | 0 |
| `--mode` | 실행 모드 (`integrated` or `detailed`) | `integrated` |
| `--output-dir` | 출력 디렉토리 | `results/<run_tag>/` |
| `--preprocess-only` | 전처리만 수행, final_metric 실행 안 함 | False |

## Python API 사용법

### 전처리만 수행

```python
from pathlib import Path
from scripts.preprocess_and_run_prod_data import preprocess_prod_data

source_csv = Path("data/prod_data_sample.csv")
processed_csv = preprocess_prod_data(
    source_csv=source_csv,
    max_rows=100,  # 처음 100행만
)
print(f"Processed: {processed_csv}")
```

### 전처리 + final_metric_refactor 실행

```python
from pathlib import Path
from scripts.preprocess_and_run_prod_data import run_final_metric_on_prod_data

source_csv = Path("data/prod_data_sample.csv")
run_final_metric_on_prod_data(
    source_csv=source_csv,
    max_rows=100,
    inspection_mode="integrated",
)
```

### 단계별 수동 처리

```python
from pathlib import Path
from scripts.preprocess_and_run_prod_data import preprocess_prod_data
from final_metric_refactor.config import FinalMetricConfig
from final_metric_refactor.run import run

# Step 1: 전처리
processed_csv = preprocess_prod_data(
    source_csv=Path("data/prod_data_sample.csv"),
    max_rows=100,
)

# Step 2: final_metric_refactor 설정
config = FinalMetricConfig(
    run_tag="my_custom_run",
    source_csv=processed_csv,
    input_col="user_context",
    output_col="assistant_response",
    label_col="eval",
    inspection_mode="detailed",
)

# Step 3: 실행
artifacts = run(config)
print(f"Results saved to: {artifacts.output_dir}")
```

## 전처리 세부사항

### user_context 추출 로직

input JSON에서 다음 순서로 시도:
1. `role == "user"`인 메시지의 `content` 추출
2. 첫 번째 메시지의 `content` 추출
3. 전체 JSON을 문자열로 반환

### assistant_response 추출 로직

output JSON에서 다음 순서로 시도:
1. `content` 필드 추출
2. `text`, `response`, `answer`, `output` 필드 중 하나 추출
3. 전체 JSON을 문자열로 반환

### 평가 점수 (eval) 추출

다음 컬럼들을 순서대로 확인하여 첫 번째 발견된 컬럼 사용:
- `correctness`
- `score`
- `A Score`
- `is_correct`
- `label`
- `eval`

없으면 기본값 "unknown"

## 출력

### 처리된 CSV 구조

기본 컬럼:
```
id,user_context,assistant_response,eval
```

선택적 컬럼 (원본 CSV에 있으면 포함):
- `traceId`
- `userId`
- `model`

### final_metric_refactor 결과

`results/<run_tag>/` 디렉토리:
- `row_results.csv` - 각 행의 평가 결과
- `summary.csv` - 요약 통계
- `rule_thresholds.csv` - 규칙별 임계값
- HTML 진단 리포트
- `run_config.json` - 실행 설정

## 예제

### 예제 1: 상세 모드로 50행만 처리

```bash
python3 scripts/preprocess_and_run_prod_data.py \
  --max-rows 50 \
  --mode detailed
```

### 예제 2: 전처리 검증

```bash
python3 scripts/preprocess_and_run_prod_data.py --preprocess-only
```

그 후 생성된 `prod_data_processed_*.csv`를 확인하여:
- `user_context`가 올바르게 추출되었는지
- `assistant_response`가 올바르게 추출되었는지
- `eval` 값이 예상과 일치하는지 확인

### 예제 3: 프로그래밍 방식으로 특정 행만 처리

```python
from pathlib import Path
import pandas as pd
from scripts.preprocess_and_run_prod_data import preprocess_prod_data

# 커스텀 필터링
df = pd.read_csv("data/prod_data_sample.csv")
filtered_df = df[df["model"] == "specific-model"].head(100)
filtered_df.to_csv("prod_data_filtered.csv", index=False)

# 전처리
processed_csv = preprocess_prod_data(
    source_csv=Path("prod_data_filtered.csv")
)
```

## 문제 해결

### 1. 평가 점수가 "unknown"인 경우

원본 CSV에 평가 컬럼이 없습니다. 스크립트를 수정하여 다른 컬럼을 사용하도록 하세요:

```python
# preprocess_and_run_prod_data.py의 preprocess_prod_data 함수 수정
eval_col = "your_custom_eval_column"
output_df["eval"] = df[eval_col]
```

### 2. user_context/assistant_response가 빈 문자열인 경우

input/output JSON 구조가 예상과 다를 수 있습니다. 다음을 확인하세요:

```python
import json
import pandas as pd

df = pd.read_csv("data/prod_data_sample.csv")
# 처음 5개 행의 input 확인
for i in range(5):
    print(f"\n--- Row {i} ---")
    print("Input structure:")
    print(json.dumps(json.loads(df.iloc[i]["input"]), indent=2)[:500])
    print("\nOutput structure:")
    print(json.dumps(json.loads(df.iloc[i]["output"]), indent=2)[:500])
```

### 3. 메모리 부족

큰 CSV를 처리할 때는 `--max-rows`를 사용하여 배치로 처리하세요:

```bash
# 배치 1
python3 scripts/preprocess_and_run_prod_data.py --max-rows 200

# 배치 2
python3 scripts/preprocess_and_run_prod_data.py --max-rows 200 --skip-rows 200
```

## 커스터마이징

전처리 로직을 커스터마이징하려면:

1. `preprocess_and_run_prod_data.py`의 `extract_user_context()` 또는 `extract_assistant_response()` 함수 수정
2. 또는 `preprocess_prod_data()` 함수를 래핑하는 커스텀 함수 작성

예:

```python
def my_custom_preprocess(source_csv):
    # 기본 전처리
    processed_csv = preprocess_prod_data(source_csv)

    # 추가 커스터마이징
    df = pd.read_csv(processed_csv)
    # ... 커스텀 로직
    df.to_csv("my_processed.csv", index=False)
    return Path("my_processed.csv")
```

## 참고

- 전체 1071행 처리 시간: 약 10-30분 (inspection_mode에 따라 다름)
- 임베딩 캐시 재구성으로 인해 첫 실행이 더 오래 걸릴 수 있음
- `--preprocess-only` 사용 시 전처리만 약 1-2초 소요
