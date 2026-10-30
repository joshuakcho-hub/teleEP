# Building the desktop application

The GitHub Actions workflow `.github/workflows/build-apps.yml` builds `TeleEP.exe` (Windows) and
`TeleEP-macOS.dmg` (macOS) whenever a version tag such as `v1.0.0` is pushed, and attaches them to the
GitHub release. To build locally:

```bash
pip install . pyinstaller
# Windows
pyinstaller --noconfirm --onefile --windowed --name TeleEP --collect-data teleep --collect-submodules teleep packaging/teleep_app.py
# macOS
pyinstaller --noconfirm --windowed --name TeleEP --collect-data teleep --collect-submodules teleep packaging/teleep_app.py
hdiutil create -volname TeleEP -srcfolder dist/TeleEP.app -ov -format UDZO TeleEP-macOS.dmg
```

The applications are not code-signed. On first launch, macOS users right-click the app and choose
Open; Windows users choose More info, then Run anyway.
