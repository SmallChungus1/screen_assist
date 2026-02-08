# screenvlm

A monolithic desktop app that captures screen, accepts user questions, and runs a VLM.

## Installation (at screen_assist directory)

```bash
pip install ".\screenvlm[rag,ui]"
```

for editable mode:

```bash
pip install -e .[rag,ui]
``` 

## Quickstart

1. Set environment variables or edit `~/.screenvlm/config.yaml`.
2. run `screenvlm run` to launch the UI.

## Configuration

See `src/screenvlm/config.py` for details.
