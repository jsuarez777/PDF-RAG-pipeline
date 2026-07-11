# Setup

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: dev/test tools
```

## System dependencies

`pdf2image` (used by `app/pdf_to_images.py` and the PDF viewer) requires
[poppler](https://poppler.freedesktop.org/) to be installed on the system —
it shells out to `pdftoppm` for the actual rendering. Without it, conversion
fails with an error like `Unable to get page count. Is poppler installed and in PATH?`

```bash
# macOS
brew install poppler

# Debian/Ubuntu
sudo apt-get install poppler-utils
```

## PDF tools

- `python app/pdf_to_images.py [file.pdf]` — convert a PDF to PNGs under
  `data/pdf2image/<YYYYMMDD_NN>/<name>/page_<n>.png` (prompts for a path if omitted).
- `python app/pdf_viewer.py` — web UI at http://127.0.0.1:5001 for uploading
  PDFs and browsing converted pages (opens your browser automatically).

Note: `data/` is gitignored; the scripts create the output folders at runtime.
