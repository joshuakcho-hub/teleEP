"""Open the TeleEP Viewer in the default browser or, with pywebview installed, in a native window."""
from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path


def viewer_file() -> str:
    base = getattr(sys, "_MEIPASS", None)          # PyInstaller bundle
    if base and os.path.exists(os.path.join(base, "TeleEP_Viewer.html")):
        return os.path.join(base, "TeleEP_Viewer.html")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "TeleEP_Viewer.html")


def open_viewer(native: bool = False) -> None:
    path = viewer_file()
    if native:
        try:
            import webview  # type: ignore
            webview.create_window("TeleEP Viewer", Path(path).as_uri(), width=1500, height=950)
            webview.start()
            return
        except ImportError:
            print("pywebview is not installed (pip install pywebview); opening in the browser instead.")
    webbrowser.open(Path(path).as_uri())


if __name__ == "__main__":
    open_viewer(native="--window" in sys.argv)
