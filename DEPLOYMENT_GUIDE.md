# Deployment Guide for v0.2.0

This guide will help you deploy version 0.2.0 to production.

## Current Status

✅ All changes committed on `improve-data-class` branch
✅ Version bumped to 0.2.0
✅ CHANGELOG.md and release notes created
✅ Documentation ready
✅ 4 example notebooks created

**Total commits on branch**: 8 commits ahead of previous state

## Step-by-Step Deployment

### Step 1: Switch to Main Branch

```bash
git checkout main
```

### Step 2: Merge improve-data-class Branch

```bash
git merge improve-data-class --no-ff -m "Merge v0.2.0 - Major feature release with Workflow API

This release includes:
- Workflow class with fluent API and parameter sweeps
- Analysis class for nbragg integration
- Complete documentation site with Sphinx
- 4 comprehensive example notebooks
- Automatic flux scaling and time range filtering
- Fixed duty cycle calculation and plot priorities

See CHANGELOG.md for full details."
```

The `--no-ff` flag ensures a merge commit is created, which is good for releases.

### Step 3: Tag the Release

```bash
git tag -a v0.2.0 -m "Release v0.2.0 - Major feature release

Highlights:
- Workflow class with parameter sweeps
- Complete documentation site
- 4 example notebooks
- Automatic flux scaling
- Enhanced plotting

See RELEASE_NOTES_v0.2.0.md for details."
```

### Step 4: Push Everything to GitHub

```bash
# Push main branch
git push origin main

# Push the tag
git push origin v0.2.0

# Push improve-data-class for history
git push origin improve-data-class
```

### Step 5: Enable GitHub Pages

1. Go to: https://github.com/TsvikiHirsh/frame_overlap/settings/pages
2. Under "Build and deployment":
   - **Source**: Select "GitHub Actions"
3. The documentation will automatically build and deploy

Monitor the build at: https://github.com/TsvikiHirsh/frame_overlap/actions

### Step 6: Create GitHub Release

1. Go to: https://github.com/TsvikiHirsh/frame_overlap/releases/new
2. **Choose a tag**: Select `v0.2.0`
3. **Release title**: `v0.2.0 - Major Feature Release`
4. **Description**: Copy contents from `RELEASE_NOTES_v0.2.0.md`
5. Click "Publish release"

### Step 7: Verify Deployment

After a few minutes, verify:

- ✅ Documentation: https://tsvikihirsh.github.io/frame_overlap/
- ✅ Release page: https://github.com/TsvikiHirsh/frame_overlap/releases/tag/v0.2.0
- ✅ Main branch updated: https://github.com/TsvikiHirsh/frame_overlap

## Post-Deployment

### Update Installation Instructions

Users can now install with:

```bash
# From GitHub (recommended until PyPI release)
pip install git+https://github.com/TsvikiHirsh/frame_overlap.git@v0.2.0

# With optional dependencies
pip install "git+https://github.com/TsvikiHirsh/frame_overlap.git@v0.2.0#egg=frame_overlap[all]"
```

### Announce the Release

Consider announcing on:
- Project README
- Documentation homepage
- Any relevant forums or mailing lists
- Social media (if applicable)

### Future PyPI Release (Optional)

If you want to publish to PyPI:

```bash
# Build distribution
python -m build

# Upload to PyPI (requires account)
python -m twine upload dist/*
```

Then users can install with:
```bash
pip install frame-overlap
```

## Troubleshooting

### If merge has conflicts:

```bash
# Abort the merge
git merge --abort

# Try again with strategy
git merge improve-data-class -s recursive -X theirs

# Or resolve conflicts manually
git merge improve-data-class
# Edit conflicting files
git add .
git commit
```

### If GitHub Pages doesn't build:

1. Check Actions tab for error messages
2. Verify `.github/workflows/docs.yml` is present on main branch
3. Ensure Pages is enabled in Settings
4. Check that pandoc installation step succeeded

### If tag already exists:

```bash
# Delete local tag
git tag -d v0.2.0

# Delete remote tag (if pushed)
git push --delete origin v0.2.0

# Recreate tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

## Summary of Changes in v0.2.0

**New Files**:
- `src/frame_overlap/workflow.py` - Workflow class
- `notebooks/example_*.ipynb` - 4 example notebooks
- `docs/` - Complete documentation structure
- `CHANGELOG.md` - Version history
- `RELEASE_NOTES_v0.2.0.md` - Release announcement
- `.github/workflows/docs.yml` - Documentation deployment

**Modified Files**:
- `src/frame_overlap/data_class.py` - Fixed plots, added pulse_duration
- `src/frame_overlap/reconstruct.py` - Added tmin/tmax, fixed __repr__
- `src/frame_overlap/analysis_nbragg.py` - Set iron as default
- `src/frame_overlap/__init__.py` - Export Workflow
- `pyproject.toml` - Version 0.2.0, updated dependencies
- `README.md` - Simplified, focused on Workflow API

**Total Additions**: ~5000+ lines
**Total Deletions**: ~500 lines

## Contact

If you encounter any issues during deployment, check:
- [Documentation](https://tsvikihirsh.github.io/frame_overlap/)
- [GitHub Issues](https://github.com/TsvikiHirsh/frame_overlap/issues)
