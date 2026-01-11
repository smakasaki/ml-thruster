# Spacecraft Thruster ML

TODO

## Project Structure

```
thruster/
├── data/
│   ├── dataset/
│   │   ├── train/          # Training data (1268 CSV files, SN01-SN12)
│   │   └── test/           # Test data (268 CSV files, SN13-SN15)
│   ├── samples/
│   │   ├── train/          # Sample training files for reference
│   │   └── test/           # Sample test files for reference
│   ├── metadata.csv        # Dataset metadata
├── pyproject.toml          # Configuration file
└── .pre-commit-config.yaml # Pre-commit hooks
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

TODO

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
