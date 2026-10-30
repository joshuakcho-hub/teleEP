# Contributing

Bug reports, questions and improvements are welcome through GitHub issues and pull requests.

## Reporting a problem

Please include the TeleEP version (`teleep --version`), the operating system, the command or steps you
used, and the full error message. If a Ponemah file does not convert, `teleep info <experiment folder>`
output helps. Do not attach unpublished animal or patient data to a public issue; describe the problem or
share a short synthetic example instead (`teleep.synthetic`).

## Development setup

```bash
git clone https://github.com/<github-account>/teleep
cd teleep
pip install -e ".[test]"
pytest
```

The viewer test (`tests/test_viewer.py`) runs only when Playwright is installed:

```bash
pip install playwright
python -m playwright install chromium
pytest tests/test_viewer.py
```

## Guidelines

* Keep the core dependencies to numpy, scipy, matplotlib and openpyxl.
* Any change to a detector must be checked against annotated data and the result reported in the pull
  request (sensitivity and positive predictivity, for example from `teleep train`).
* Any change to an HRV formula must be made in both `src/teleep/hrv.py` and the viewer
  (`src/teleep/viewer/TeleEP_Viewer.html`); `tests/test_viewer.py` checks that they agree.
