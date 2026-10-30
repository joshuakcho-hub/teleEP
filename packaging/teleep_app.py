"""Entry point for the packaged desktop application (PyInstaller)."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from teleep.gui.app import main
    main()
