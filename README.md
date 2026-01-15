# Spacecraft Thruster ML

TODO

## Project Structure

```
thruster/
├── data/
│   ├── dataset/
│   │   ├── metadata.csv    # Dataset metadata
│   │   ├── train/          # Training data (SN01-SN12)
│   │   └── test/           # Test data (SN13-SN15)
│   ├── samples/
│   │   ├── metadata.csv    # Sample metadata
│   │   ├── train/          # Sample training files for reference
│   │   └── test/           # Sample test files for reference
│   ├──
├── pyproject.toml          # Configuration file
└── .pre-commit-config.yaml # Pre-commit hooks
...
```

## Dataset

TODO

## Setup

### Prerequisites

- Python 3.11+

### Installation

1. Clone the repository.

2. Create virtual environment:
```bash
python -m venv venv
```

3. Activate virtual environment:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -e .
```

Or install with dev tools:
```bash
pip install -e ".[dev]"
```

## Usage

Run full pipeline:
```bash
python -m src.scripts.first_midterm
```

Run with options:
```bash
python -m src.scripts.first_midterm --log-file --save-cleaned-csv
```

Run interactive menu:
```bash
python -m src.scripts.first_midterm --interactive
```

### Command-line Arguments

- `--log-file` - Enable logging to file (outputs/pipeline.log)
- `--save-cleaned-csv` - Save individual cleaned CSV files
- `--interactive` - Launch interactive menu

### Interactive Menu

**[1] Run Full Pipeline** - Complete workflow: load, clean, aggregate, statistics, visualizations

**[2] Load and Clean Data** - Process metadata and time series files

**[3] Generate Statistics** - Create statistical reports and tables

**[4] Create Visualizations** - Generate all plots and figures

**[5] Run Statistics + Visualizations** - Execute options 3 and 4

**[0] Exit** - Close the application

## Development

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

Run hooks manually:
```bash
pre-commit run --all-files
```

Update hooks to latest versions:
```bash
pre-commit autoupdate
```

### Manual Checks

Run linter:
```bash
ruff check .
```

Format code:
```bash
ruff format .
```

Type checking:
```bash
pyright
```
