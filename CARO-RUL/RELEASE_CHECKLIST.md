# Before making the repository public

1. Confirm the authors/institution authorize redistribution of processed SW1–SW7 data, predictions, model weights and provenance metadata. The original dataset metadata says redistribution is not cleared; do not publish these materials until the rights holders approve. Retain or update the data-use terms with their permission.
2. Confirm a license for the authors’ software; preserve the vendor LICENSE files.
3. Unzip this package into the existing `CARO-RUL/` directory of https://github.com/2xinxin/CARO-RUL . Keep the folder layout. Do not upload private raw waveform archives.
4. Commit the files and create a `v1.0.0` tag/release. Version 1.0.0 in the local package is not a claim that the GitHub release already exists.
5. Check access anonymously, download a fresh copy, and run `python scripts/verify_package.py`. Follow README reproduction commands.
6. Only after these steps use the GitHub-based manuscript availability statement in the publication files. Prefer replacing its main-branch URL with the corresponding verified fixed tag or commit URL.
