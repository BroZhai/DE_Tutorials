# CS5481 Tutorial 3: Pre-class Environment Setup and Live Coding Protocol

## Scope

This tutorial uses Python, Jupyter Notebook, and pandas. Choose either of the following ways to run the code during class:

1. A local environment in VS Code, with files stored on your computer; or
2. Google Colab, which runs in a browser and does not require a local Python installation.

Please allow at least 15-20 minutes before class to complete the setup and test it. You are ready to follow the tutorial once the pre-class check at the end displays `Ready`.

## 1. Files you need

Download the student materials from the course platform and keep the filenames unchanged.

| File | Purpose |
|---|---|
| `tutorial3_week3.ipynb` |  Main notebook used during class |
| `movie_metadata.csv` | Raw movie dataset and input to the notebook |


Google Colab filenames are case-sensitive. Do not rename `movie_metadata.csv` to `Movie_Metadata.csv`, and make sure your browser has not renamed it to something such as `movie_metadata (1).csv`.

---

## 2. Option A: Local environment in VS Code

### 2.1 Install the required software

Install the following before class:

- Python 3.10, 3.11, or 3.12;
- Visual Studio Code;
- These VS Code extensions:
  - **Python** by Microsoft;
  - **Jupyter** by Microsoft.

Windows users are advised to select **Add Python to PATH** when installing Python. If Python is already installed, check it in a terminal:

```bash
python --version
```

On macOS or Linux, use the following if the `python` command is unavailable:

```bash
python3 --version
```

### 2.2 Create a tutorial folder

Create a separate folder, for example:

```text
CS5481_Tutorial3/
├── tutorial_week3.ipynb
└── movie_metadata.csv
```

It is best to use only English letters, numbers, underscores, and hyphens in the folder path. Avoid cloud-synchronised directories, very long paths, and special characters when possible.

The notebook and `movie_metadata.csv` must be in the same folder because the notebook uses a relative path:

```python
pd.read_csv("movie_metadata.csv", encoding="utf-8")
```

### 2.3 Open the entire folder in VS Code

1. Start VS Code;
2. Select **File > Open Folder...**;
3. Open the `CS5481_Tutorial3` folder created above;
4. Do not open only the notebook by double-clicking it, because the working directory may then differ from the data directory;
5. Open `tutorial3_week3.ipynb` from inside VS Code.

### 2.4 Create an isolated Python environment

We recommend creating a `.venv` environment for this tutorial. This prevents packages from other courses or projects from interfering with the tutorial.

#### Windows PowerShell

In VS Code, select **Terminal > New Terminal**. Confirm that the terminal is inside the `CS5481_Tutorial3` folder, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas jupyter ipykernel
```

If PowerShell does not allow the activation script to run, do not accidentally install the packages into the system Python. Instead, run:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install pandas jupyter ipykernel
```

You can then select the `.venv` interpreter directly as described in the next section.

#### macOS / Linux

Run the following commands in the VS Code terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas jupyter ipykernel
```

After installation, test pandas with:

```bash
python -c "import pandas as pd; print(pd.__version__)"
```

If this prints a version number without an error, pandas is installed correctly.

### 2.5 Select the correct notebook kernel

1. Open `tutorial3_week3.ipynb`;
2. Click **Select Kernel** in the top-right corner;
3. Select **Python Environments**;
4. Select the `.venv` environment you just created;
5. Allow VS Code to install or enable the required extensions if prompted;
6. Run the pre-class check in Section 4 below.

The Python environment used to install packages in the terminal must be the same environment selected as the notebook kernel. If pandas is installed in the terminal but the notebook still raises `ModuleNotFoundError`, the wrong kernel has probably been selected.

### 2.6 Where local output is saved

The following line in the main notebook:

```python
data.to_csv("cleanfile.csv", encoding="utf-8")
```

creates the output in the current working directory:

```text
CS5481_Tutorial3/
├── tutorial_week3.ipynb
├── movie_metadata.csv
└── cleanfile.csv          # Created after the save cell is run
```

Running the same cell again overwrites the existing `cleanfile.csv`. The original `movie_metadata.csv` is not overwritten.

---

## 3. Option B: Google Colab

Google Colab provides Python and pandas, so you normally do not need to install software on your computer. You can use Chrome, Edge, Safari, or Firefox.

### 3.1 Open the notebook

1. Open [Google Colab](https://colab.research.google.com/);
2. Sign in to your Google account;
3. Select **Upload** in the welcome window;
4. Upload `tutorial3_week3.ipynb`;
5. Wait for the notebook to open and connect to a runtime.

The notebook contains the following cell:

```python
!pip install pandas
```

You may run it in Colab, although pandas is normally already installed. If Colab asks you to restart the runtime after installation, restart it and begin again from the **Import Libraries** cell.

### 3.2 Method B1: Upload the data temporarily (simplest for class)

Use this method if you only need the files for one class. The data is stored in `/content` for the current Colab session. You must upload it again if the runtime is reset or disconnected.

#### Upload through the Files panel

1. Click the folder icon on the left side of Colab;
2. Click **Upload to session storage**;
3. Upload `movie_metadata.csv`; no other CSV is required;
4. Wait for the upload to finish and confirm that the complete filename appears in the file list.

Alternatively, create and run a new code cell:

```python
from google.colab import files
uploaded = files.upload()  # Select movie_metadata.csv in the file picker
```

After uploading, run:

```python
from pathlib import Path

print("Current folder:", Path.cwd())
print("CSV exists:", Path("movie_metadata.csv").is_file())
print("CSV size:", Path("movie_metadata.csv").stat().st_size, "bytes")
```

The output should include:

```text
Current folder: /content
CSV exists: True
```

You can then use the existing notebook code without changing the path:

```python
data = pd.read_csv("movie_metadata.csv", encoding="utf-8")
```

#### Download the generated cleanfile.csv

After running the save cell, find `cleanfile.csv` in the Files panel. Click the three dots next to the file and select **Download**. You can also run:

```python
from google.colab import files
files.download("cleanfile.csv")
```

Download any results you want to keep before closing Colab. Files in `/content` disappear when the runtime ends.

### 3.3 Method B2: Use Google Drive (persistent storage)

Use this method if you want the files to remain available the next time you open the notebook.

#### Prepare the files in Google Drive

1. Under **My Drive**, create the folder `CS5481/Tutorial3`;
2. Upload `tutorial3_week3.ipynb` and `movie_metadata.csv` to that folder;
3. Open the notebook in Colab;
4. Add a code cell at the very beginning and run:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Your browser will ask you to authorise Colab to access Google Drive. After authorisation, run:

```python
from pathlib import Path
import os

PROJECT_DIR = Path("/content/drive/MyDrive/CS5481/Tutorial3")
assert PROJECT_DIR.is_dir(), f"Folder not found: {PROJECT_DIR}"
assert (PROJECT_DIR / "movie_metadata.csv").is_file(), "movie_metadata.csv not found"

os.chdir(PROJECT_DIR)
print("Current folder:", Path.cwd())
```

Important: opening a notebook from Google Drive does not necessarily make its Drive folder the Colab working directory. You must run `os.chdir(PROJECT_DIR)` as shown above, or provide the full Drive path in `pd.read_csv()`.

After changing directories, the existing relative-path code works without modification:

```python
data = pd.read_csv("movie_metadata.csv", encoding="utf-8")
```

The generated `cleanfile.csv` will now be saved directly in the `CS5481/Tutorial3` folder in Google Drive and will not disappear when the Colab runtime ends.

If you used a different Drive folder name, update `PROJECT_DIR` accordingly. Avoid combining temporary upload and Google Drive in the same session, because you may accidentally read the wrong copy of the CSV file.

---

## 4. Pre-class check for both options

Temporarily add a new code cell near the top of the main notebook and run:

```python
from pathlib import Path
import sys
import pandas as pd

csv_path = Path("movie_metadata.csv")

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("Working directory:", Path.cwd())
print("CSV path:", csv_path.resolve())

assert csv_path.is_file(), "Cannot find movie_metadata.csv; check its name and the working directory"

test_data = pd.read_csv(csv_path, encoding="utf-8")
assert test_data.shape == (5043, 28), f"Unexpected CSV file or content: {test_data.shape}"

print("Ready - movie_metadata.csv loaded successfully:", test_data.shape)
```

The final line of the correct output is:

```text
Ready - movie_metadata.csv loaded successfully: (5043, 28)
```

Once you see `Ready`, save the notebook. You do not need to run the later demonstration cells before class.

---

## 5. Common problems and quick fixes

### `FileNotFoundError: movie_metadata.csv`

The file is usually missing from the current working directory or has been renamed. Check with:

```python
from pathlib import Path
print(Path.cwd())
print([p.name for p in Path.cwd().iterdir()])
```

- VS Code: use **Open Folder** to open the folder that contains both the notebook and the CSV;
- Colab temporary upload: upload the CSV to `/content` again;
- Colab with Drive: mount Drive again and rerun `os.chdir(PROJECT_DIR)`;
- Check whether the file was renamed to `movie_metadata (1).csv`.

### `ModuleNotFoundError: No module named 'pandas'`

- VS Code: confirm that the selected kernel is `.venv`, then run `python -m pip install pandas` in that environment;
- Colab: run `%pip install pandas` in a cell, then restart the runtime if prompted.

### `NameError: name 'data' is not defined`

A required earlier cell has not been run, or the kernel has restarted. Begin again with `import pandas as pd` and `pd.read_csv(...)`, then run the cells in order.

### `SyntaxError` in the Practice section

The exercise answers are still blank. Complete the expressions on the right-hand side of the equals signs before running the cell. The environment itself is probably working correctly.

### Files disappear after Colab reconnects

Files uploaded temporarily to `/content` are not permanent. Upload `movie_metadata.csv` again or use the Google Drive method.

### cleanfile.csv has an extra `Unnamed: 0` column

The current notebook uses `data.to_csv("cleanfile.csv", encoding="utf-8")`. By default, pandas writes the DataFrame index to the CSV, so reading the output back creates an additional column. This is the expected result of the current code. To omit the index, use:

```python
data.to_csv("cleanfile.csv", encoding="utf-8", index=False)
```

### An `invalid escape sequence` warning appears in a regular-expression cell

Some Python versions warn that `\s` is not an ordinary string escape. The expression will generally still run, but raw strings are recommended for regular expressions:

```python
re.search(r"\s", txt)
re.split(r"\s", txt)
re.sub(r"\s", "9", txt)
```

---

