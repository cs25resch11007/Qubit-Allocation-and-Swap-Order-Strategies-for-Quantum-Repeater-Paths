# Entanglement Swapping Algorithms and Simulation

This folder contains three Python scripts for studying entanglement swapping orders,
qubit/memory allocation, and simulated throughput in quantum repeater path networks.

## Files

### `ProposedEntanglementSwappingAlgorithms.py`

Implements entanglement-swapping optimization algorithms and supporting probability
distribution routines. The code includes dynamic programming and heuristic approaches,
binomial-distribution calculations, and traditional/vectorized Bell-state-measurement
(BSM) convolution engines. The `USE_VECTORIZED_CONV` flag selects the vectorized
convolution implementation when enabled.

Run it according to the entry point and arguments defined in the script:

```bash
python ProposedEntanglementSwappingAlgorithms.py --help
```

If the script does not define a command-line parser in your version, inspect the
bottom of the file for its `main`/experiment entry point and configuration variables.

### `entanglement_swapping.py`

Simulates asynchronous entanglement generation and entanglement swapping on a path
network using SeQUeNCe. It supports named swap orders (`vora`, `baln`, `bal2`, `L2R`,
and `R2L`) as well as explicit orders, configurable link distances, memory allocation,
coherence time, swap success probabilities, and repeated runs.

Examples:

```bash
# Run all named orders
python entanglement_swapping.py

# Run one named order
python entanglement_swapping.py --order vora

# Run an explicit order
python entanglement_swapping.py --order 1 3 2 4

# Average each named order over five runs
python entanglement_swapping.py --runs 5

# Set allocation and coherence time
python entanglement_swapping.py --alloc 3 3 3 3 3 --coherence 10

# Use custom link distances
python entanglement_swapping.py --distances 50 50 50 --alloc 3 3 3 --order 1 2

# Use the literal simulation parameters rather than the calibrated settings
python entanglement_swapping.py --literal
```

Check available options with `python entanglement_swapping.py --help`.

### `run_regime_average.py`

Runs repeated throughput experiments for algorithms and regimes described by worksheets
such as `Wide Random`, `High Fidelity`, `Mid-Noise`, and `Random Bottleneck 30%`.
It writes an Excel summary and a CSV log of individual runs, allowing completed runs to
be reused if a batch is interrupted.

**Additional project files are required:** this script imports `run_5node_cases.py`
and reads an Excel workbook. Those files are not included in the three scripts supplied
here. Place them in the expected location or specify the workbook path with `--workbook`.

Examples (once the supporting files are available):

```bash
# Dry-run: list cases without running simulations
python run_regime_average.py --all-cases --dry-run

# One selected/random case per regime; 30 runs per case by default
python run_regime_average.py --jobs 4

# Every case in each worksheet, one run per case by default
python run_regime_average.py --all-cases --jobs 4

# Every case, three runs each
python run_regime_average.py --all-cases --runs 3 --jobs 4

# Select a particular case in one regime
python run_regime_average.py --case "Wide Random=3" --jobs 4

# Supply an explicit workbook path
python run_regime_average.py --workbook path/to/testcases.xlsx --all-cases --dry-run
```

Use `python run_regime_average.py --help` to see all options.

## Requirements

- Python 3.10 or newer is recommended.
- Python packages listed in `requirements.txt`.
- SeQUeNCe for `entanglement_swapping.py`.
- The project's `sequence` package/source tree must be importable by Python. The
  script adjusts `sys.path` for a local project layout, but it does not install
  SeQUeNCe automatically.
- `run_5node_cases.py` and the input Excel workbook are needed to execute
  `run_regime_average.py`.

## Setup

Create and activate a virtual environment.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install and configure SeQUeNCe separately according to the version/repository used by
your project. Verify that `from sequence.kernel.timeline import Timeline` works before
running `entanglement_swapping.py`.

## Suggested folder layout

Keep the three scripts together. For the regime-average runner, also provide the
supporting runner module and workbook, for example:

```text
project/
├── ProposedEntanglementSwappingAlgorithms.py
├── entanglement_swapping.py
├── run_regime_average.py
├── run_5node_cases.py       # required by run_regime_average.py;
├── testcases.xlsx           # example workbook; use your actual workbook
└── requirements.txt
```

The exact workbook name and location depend on your project configuration.

## Outputs

- `entanglement_swapping.py` prints run results and throughput statistics to the terminal.
- `run_regime_average.py` produces an Excel summary (including an `Averages` sheet and,
  in all-cases mode, a `Per case` sheet) plus a CSV of individual runs. Default output
  paths are derived from the workbook path; `--out` and `--runs-csv` override them.

## Notes

- Simulations can be computationally expensive. Start with a small number of runs/jobs
  before launching large batches.
- Results depend on the simulation parameters, SeQUeNCe version, input workbook, random
  seeds, and machine resources.
- The requirements file lists external Python packages imported by the supplied code.
  Python standard-library modules do not need to be installed with `pip`.
