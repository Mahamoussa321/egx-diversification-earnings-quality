# EGX Corporate Diversification and Earnings Quality

This repository provides reproducible research code for studying the relationship between corporate diversification, information asymmetry, corporate governance quality, and earnings quality among firms listed on the Egyptian Exchange (EGX).

The empirical framework examines whether diversification affects earnings quality directly and indirectly through information asymmetry, and whether corporate governance quality moderates the information-asymmetry channel.

## Research Framework

```text
Corporate Diversification → Information Asymmetry → Earnings Quality
                                      ↑
                         Corporate Governance Quality
```

## Hypotheses

- **H1:** Corporate diversification significantly affects earnings quality.
- **H2:** Corporate diversification significantly affects information asymmetry.
- **H3:** Information asymmetry significantly affects earnings quality.
- **H4:** Information asymmetry mediates the relationship between corporate diversification and earnings quality.
- **H5:** Corporate governance quality moderates the relationship between information asymmetry and earnings quality.
- **H6:** The indirect effect of corporate diversification on earnings quality through information asymmetry varies according to corporate governance quality.

## Repository Contents

```text
config/                 Variable mappings and analysis settings
src/                    Reusable Python functions
scripts/                Reproducible command-line scripts
stata/                  Optional Stata analysis file
docs/                   Methods notes, checklist, and research-model figure
notebooks/              Optional exploratory notebook folder
outputs/tables/         Generated descriptive and regression tables
outputs/figures/        Generated manuscript figures
data/raw/               Private raw data for local reproduction; ignored by Git
data/processed/         Processed analysis panel; ignored by Git
```

## Data Availability and Privacy

The public GitHub repository is designed to share the code, documentation, and reproducible output structure. Raw data files are not intended for public distribution.

For local reproduction, authorized users should place the raw Excel files in:

```text
data/raw/
```

The local archive version of this project may include private raw data for long-term preservation, but `.gitignore` prevents these files from being committed to GitHub.

Ignored private data locations include:

```text
data/raw/
data/processed/
```

This protects raw financial, governance, and trading data from being uploaded publicly.

## Required Raw Data Files for Full Reproduction

The current pipeline expects the following local files:

```text
data/raw/egx_diversification_existing_data.xlsx
data/raw/egx_governance_variables.xlsx
data/raw/Amihud - Stock Liquidity.xlsx
```

The Amihud workbook is used to construct annual information asymmetry from daily stock liquidity data.

## Setup

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducible Workflow

### 1. Inspect the input workbooks

```bash
python scripts/00_inspect_excel.py
```

This creates:

```text
outputs/tables/excel_inventory.csv
```

### 2. Prepare the firm-year panel

```bash
python scripts/01_prepare_panel.py
```

This creates:

```text
data/processed/analysis_panel.csv
```

### 3. Run the main econometric analysis

```bash
python scripts/02_run_econometrics.py
```

This creates descriptive, correlation, VIF, and regression tables in:

```text
outputs/tables/
```

### 4. Run the full pipeline

```bash
python scripts/run_all.py
```

### 5. Optional machine-learning robustness

```bash
python scripts/03_run_ml_robustness.py
```

The machine-learning component is intended for robustness and predictive validation. It should not replace the main econometric hypothesis tests.

## Empirical Design

The main analysis uses firm-year panel models with controls and fixed effects when feasible.

Direct effect model:

```text
EQ_it = beta_0 + beta_1 DIV_it + Controls_it + Firm FE + Year FE + epsilon_it
```

Information-asymmetry model:

```text
IA_it = alpha_0 + alpha_1 DIV_it + Controls_it + Firm FE + Year FE + epsilon_it
```

Moderated mediation model:

```text
EQ_it = gamma_0 + gamma_1 DIV_it + gamma_2 IA_it + gamma_3 CGQ_it
        + gamma_4 IA_it x CGQ_it + Controls_it
        + Firm FE + Year FE + epsilon_it
```

Conditional indirect effect:

```text
Indirect effect = alpha_1 x (gamma_2 + gamma_4 x CGQ)
```

## Main Variables

| Symbol | Description |
|---|---|
| `EQ` | Earnings quality |
| `DA` | Discretionary accruals |
| `DIV` | Corporate diversification |
| `AMIHUD` | Amihud illiquidity / information asymmetry |
| `CGQ` | Corporate governance quality index |
| `ROA` | Return on assets |
| `ROE` | Return on equity |
| `Leverage` | Total debt divided by total assets |
| `size` | Firm size |
| `age` | Firm age |
| `Sales Growth` | Annual sales growth |

## Reproducibility Notes

The code includes a transparent fallback diversification proxy based on available branch and production-line information when segment-sales data are unavailable. For journal submission, a sales-based Herfindahl or entropy measure is preferred when segment-sales data can be obtained.

Amihud illiquidity is constructed from daily stock data as the annual mean of absolute daily return divided by trading value, with scaling and transformation handled in the data-preparation script.

## GitHub Sharing Policy

This repository should publicly share:

- code in `scripts/` and `src/`,
- configuration files,
- reproducibility documentation,
- curated result tables and figures.

This repository should not publicly share:

- raw Excel data,
- processed private panels,
- manuscript drafts,
- local environment files.

The `.gitignore` file is configured accordingly.

## License

This project is provided for academic and research use. See `LICENSE` for details.
