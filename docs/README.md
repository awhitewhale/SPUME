# SPUME project page

This directory contains the static project page published at
`https://awhitewhale.github.io/SPUME/`.

The page is intentionally build-free: serve `docs/` with any static HTTP
server. Web-ready method figures, observation/SPUME reveal panels, and the
interactive Water3D point clouds can be regenerated without modifying their
sources:

```powershell
python tools/prepare_site_assets.py `
  --manuscript-dir "E:\path\to\SPUME_TIP_Final_Revised_Manuscript_Ready_for_Review" `
  --water3d-root "F:\underwater3D\Water3D" `
  --site-root "docs" `
  --pdftoppm "C:\path\to\pdftoppm.exe"
```

Only the compressed project-page assets are versioned. Full-resolution
Water3D outputs remain outside the repository.
