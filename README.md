# Spectrogram Matcher

A PySide6 GUI tool for interactively matching spectrograms.
The tool:

- Loads spectrograms and their embeddings from an HDF5 file.
- Computes pairwise cosine distances between all spectrogram embeddings.
- Lets you pick a **query spectrogram**.
- Shows two proposal sets:
  1. **20 nearest neighbours** (lowest cosine distance)
  2. **20 quantile-ranked** examples from the query’s distance distribution
- Lets you label each proposal as **Match** or **No match**.
- Lets you save your annotations to a JSON file.

---

## 📦 Installation

1. Clone or download this repository:

```bash
git clone https://github.com/janclemenslab/spectrogram-matcher.git
cd spectrogram-matcher
```

2. Install dependencies:

```bash
conda create -n sm -y python=3.13 PySide6 numpy h5py matplotlib -c conda-forge
```

2. Download the data from this [link](https://www.dropbox.com/scl/fi/gsgrppkc91xee5i5x0klf/embeddings.h5?rlkey=k0tmb5pj6myg4i7ju5mj36uzc&dl=0) and put it in the same folder as the file `spectrogram_matcher.py`.



## ▶️ Usage

Change into the directory with the `spectrogram_matcher.py` and `embeddings.h5` file and run the application:

```bash
conda activate sm
python spectrogram_matcher.py --annotator YOURNAME
```


## ⌨️ Shortcuts

- **Ctrl+S** — Save results
- **N** — Next query
- **R** — Pick random query

---

Created with ChatGPT5