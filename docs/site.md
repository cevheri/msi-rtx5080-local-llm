# The model guide: how it is built and how to change it

Last verified: 2026-11-19.

The [model guide](https://cevheri.github.io/msi-rtx5080-local-llm/) shows the measurements and agent test results recorded in this repository.
It is a static site: plain HTML files with no server behind them, built from the repository's own files.
GitHub Actions ([`.github/workflows/pages.yml`](../.github/workflows/pages.yml)) runs the tests, builds the site and publishes it on every push to `main`.

## How it is built

```text
Repository data                launcher/site/ (Python)                            Output (_site/)
recipes/*.toml           --+
docs/model-selection.md  --+--> data.py --> fit.py --> words.py --+--> render.py --> index.html
research/*/*/summary.md  --+    (read)      (decide)   (phrase)   |    (HTML)        models/<id>/index.html
README.md, host-setup.md --+                                      |
                                draw.py  (memory drawings) -------+
                                state.py (precomputed answers) ---+
```

`data.py` reads the repository and stops the build with a message that names the file when something is malformed.
`fit.py` decides which setup fits which GPU and RAM, what the page recommends and how the lists sort.
`words.py` holds every sentence the page shows.
`render.py` writes the home and one page per model.
`state.py` computes the answer for every GPU, RAM and agent choice in advance and puts it into each page.
In the browser, `page.js` only switches between those answers when the reader clicks; it computes nothing and calls no server.

## Where each part lives

| To change | Edit |
|---|---|
| A measurement, a model or the ranking | The data: `recipes/`, `docs/model-selection.md`, `research/`; the site code stays as it is |
| A sentence or a label on the page | `launcher/site/words.py` |
| Colours, spacing, fonts, layout | `launcher/site/page.css` (the colour tokens are at its top) |
| The order or presence of a block | `launcher/site/render.py` |
| Which setup fits a machine, the recommendation, the sort orders | `launcher/site/fit.py` |
| The two estimates (GPU memory for the desktop, the share of RAM that is free) | The constants at the top of `launcher/site/fit.py` |
| What the page reads from the repository | `launcher/site/data.py` |
| What changes when the reader clicks | `launcher/site/state.py` and `launcher/site/page.js` |

## Preview a change

`heavy python3 -m launcher.preview` builds the site from this checkout, serves it at http://127.0.0.1:8700/ and rebuilds it about a second after any change to the code or the data; reload the browser to see it.
A broken build prints its message and the preview keeps showing the last build that worked.
Ctrl+C stops it.

## Check a change

`python3 -m unittest discover -s launcher/tests -t .` runs every test in under half a minute.
Some tests pin sentences word for word (`launcher/tests/test_site_words.py`), so a wording change also updates the test beside it.
A change that passes the tests and looks right in the preview is ready for a pull request; CI runs the same tests on Python 3.11 before the site is published.
