# ppg

## Files

- `receiver.py`
  - Receives Galaxy Watch UDP data
  - Saves `raw_*.csv`
  - Saves `processed_*.csv`
  - Saves `dump_*.txt`

- `pipeline.py`
  - Called internally by `receiver.py`
  - Performs PPG preprocessing
  - Generates 1250-sample windows in the input format required by PaPaGEI

- `papagei-s, papagei-p.py` — embedding extraction only
  - Loads `processed_*.csv`
  - Loads the PaPaGEI model
  - Extracts embeddings
  - Saves them to `.npz` files

## PaPaGEI installation

- Clone the PaPaGEI repository

```bash
git clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model.git
```

- Move into the PaPaGEI folder

```bash
cd papagei-foundation-model
```

- Create a conda environment

```bash
conda create -n papagei_env python=3.10
conda activate papagei_env
```

- Install dependencies

```bash
pip install -r requirements.txt
pip install pyPPG==1.0.41
```

## Prepare weights

- Download the PaPaGEI weights
- Save them in the `weights/` folder

```text
weights/papagei_p.pt
weights/papagei_s.pt
```

- `papagei_p.pt`
  - Used when `MODEL_TYPE = "p"`

- `papagei_s.pt`
  - Used when `MODEL_TYPE = "s"`

## Run receiver

- Receive Watch data

```bash
python receiver.py --port 5005
```

- Files generated after execution

```text
raw_YYYYMMDD_HHMMSS.csv
processed_YYYYMMDD_HHMMSS.csv
dump_YYYYMMDD_HHMMSS.txt
```

- Only `processed_*.csv` is used for embedding extraction

## Run embedding extraction

- Update the paths at the top of `papagei-p` and `papagei-s`

- Run

```bash
python papagei-p.py
or
python papagei-s.py
```

## Output

- Generates the `.npy` file specified by `OUTPUT`

- Stored value

```text
embeddings
```

- `embeddings`
  - PaPaGEI embedding
  - shape: `(N, 512)`

## Data

- `processed_*.csv` — preprocessed data
- `raw_*.csv` — data before running the pipeline
- `dump_*.txt` — original raw dump
