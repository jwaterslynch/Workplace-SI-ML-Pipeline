# Release Checklist

Use this before tagging a GitHub release or citing the tool on the website.

## Repository Hygiene

- [ ] Confirm the working tree contains only intentional changes.
- [ ] Confirm raw downloaded NSDUH files are not tracked.
- [ ] Confirm no local paths, secrets, credentials, or private notes are exposed.
- [ ] Confirm large output artifacts are intentional and documented.

## Reproducibility

- [ ] Run `./code/run_si.sh verify`.
- [ ] For model or data-construction changes, run `./bootstrap.sh`.
- [ ] Confirm `outputs/metadata.json` reflects the release environment.
- [ ] Confirm `data/checksums.txt` is current after a full rerun.
- [ ] Confirm headline metrics match `README.md` tolerances.

## Documentation

- [ ] Check `README.md` quick-start commands.
- [ ] Check `OSF_README.md`.
- [ ] Check `docs/SAFETY_AND_USE_BOUNDARIES.md`.
- [ ] Check `docs/WEBSITE_CARD.md`.
- [ ] Update `CHANGELOG.md`.
- [ ] Update `CITATION.cff` version and release date.

## Website And Archival Links

- [ ] Decide whether to rename the GitHub repository to a non-workplace slug.
- [ ] Confirm GitHub repository URL is correct.
- [ ] Confirm OSF URL is correct.
- [ ] Add or update the Research Tools card on `julianwaterslynch.com`.
- [ ] Link the tool from the related paper entry.
- [ ] Archive the release if assigning a DOI through OSF, Zenodo, or GitHub.

## Final

- [ ] Tag the release, for example `v0.1.0`.
- [ ] Publish GitHub release notes.
- [ ] Confirm GitHub displays the citation panel from `CITATION.cff`.
- [ ] Confirm the website card points to the released version.
