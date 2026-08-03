# ppg


## 파일 구성

- `receiver.py`
  - Galaxy Watch UDP 데이터 수신
  - `raw_*.csv` 저장
  - `processed_*.csv` 저장
  - `dump_*.txt` 저장

- `pipeline.py`
  - `receiver.py` 내부에서 호출
  - PPG 전처리 수행
  - PaPaGEI 입력 형태인 1250-sample window 생성

- `papagei-s, papagei-p.py` - embedding 추출만 진행
  - `processed_*.csv` 로드
  - PaPaGEI 모델 로드
  - embedding 추출
  - `.npz` 파일로 저장
 
    

## PaPaGEI 설치

- PaPaGEI repository 다운로드

```bash
git clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model.git
```

- PaPaGEI 폴더로 이동

```bash
cd papagei-foundation-model
```

- conda 환경 생성

```bash
conda create -n papagei_env python=3.10
conda activate papagei_env
```

- dependency 설치

```bash
pip install -r requirements.txt
pip install pyPPG==1.0.41
```

## Weight 준비

- PaPaGEI weight 다운로드
- `weights/` 폴더에 저장

```text
weights/papagei_p.pt
weights/papagei_s.pt
```

- `papagei_p.pt`
  - `MODEL_TYPE = "p"`일 때 사용

- `papagei_s.pt`
  - `MODEL_TYPE = "s"`일 때 사용

## Receiver 실행

- Watch 데이터 수신

```bash
python receiver.py --port 5005
```

- 실행 후 생성 파일

```text
raw_YYYYMMDD_HHMMSS.csv
processed_YYYYMMDD_HHMMSS.csv
dump_YYYYMMDD_HHMMSS.txt
```

- embedding 시에는 `processed_*.csv`만 사용

## Embedding 실행

- `papagei-p, papagei-s` 상단 경로 수정

- 실행

```bash
python papagei-p.py
혹은
python papagei-s.py
```

## Output

- `OUTPUT`에 지정한 `.npy` 파일 생성

- 저장 값

```text
embeddings
```

- `embeddings`
  - PaPaGEI embedding
  - shape: `(N, 512)`

## 데이터 

- `processed_*.csv` - 전처리 완료
- `raw_*.csv` - pipeline 실행 전
- `dump_*.txt` - 원본 raw


## 데이터 

- `raw_20260414_191218.csv` - 팔 강하게 흔들면서 수집
- `raw_20260414_190911.csv`	- 가만히 있다가 약간씩 흔들면서 수집
- `raw_20260413_113425.csv` - 보행하며 수집

추가적으로 필요한 데이터 있으면 바로 수집해서 넣어 두도록 하겠습니다!
- 
