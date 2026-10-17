# Streamlit Cloud Deployment Guide

This guide explains how to deploy the frame_overlap app to Streamlit Community Cloud and resolve nbragg installation issues.

## The nbragg Dependency Issue

### Problem
`nbragg` depends on `NCrystal`, which is a C++ library that needs to be compiled during installation. Streamlit Cloud's default environment doesn't include the necessary build tools (gcc, g++, cmake), causing the installation to fail with:

```
nbragg fit failed: nbragg is required for Analysis class. Install with: pip install nbragg
```

### Solution
We provide a `packages.txt` file that tells Streamlit Cloud to install the necessary system packages before installing Python packages.

## Required Files for Deployment

### 1. `packages.txt`
This file contains system-level dependencies needed for NCrystal compilation:

```
build-essential
cmake
g++
python3-dev
```

**Location:** Root directory of the repository

**Purpose:** Streamlit Cloud reads this file and installs these Ubuntu packages before installing Python requirements.

### 2. `requirements.txt`
Standard Python dependencies:

```
streamlit>=1.28.0
plotly>=5.17.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-image>=0.19.0
tqdm>=4.60.0
lmfit>=1.0.0
nbragg>=0.5.0
```

**Location:** Root directory of the repository

**Note:** `nbragg>=0.5.0` will automatically install `ncrystal` as a dependency, which will be compiled using the build tools from `packages.txt`.

### 3. `.streamlit/config.toml` (Optional)
Streamlit configuration for theme and server settings:

```toml
[theme]
primaryColor = "#3384beff"
backgroundColor = "#dcdcdcff"
secondaryBackgroundColor = "#c1cbe0ff"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
enableCORS = true
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

## Deployment Steps

### Initial Setup

1. **Push your code to GitHub:**
   ```bash
   git add packages.txt requirements.txt
   git commit -m "Add packages.txt for NCrystal compilation on Streamlit Cloud"
   git push origin main
   ```

2. **Go to Streamlit Community Cloud:**
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy the app:**
   - Click "New app"
   - Select your repository: `TsvikiHirsh/frame_overlap`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"

### During Deployment

Streamlit Cloud will:
1. Read `packages.txt` and install system packages (build-essential, cmake, g++, python3-dev)
2. Read `requirements.txt` and install Python packages
3. When installing `nbragg`, it will compile NCrystal using the installed build tools
4. Start the Streamlit app

**Note:** The first deployment may take 5-10 minutes due to NCrystal compilation.

### Troubleshooting

#### If deployment fails with nbragg import error:

1. **Check that `packages.txt` exists in the root directory**
   ```bash
   ls -la packages.txt
   ```

2. **Verify packages.txt contents:**
   ```bash
   cat packages.txt
   ```
   Should contain:
   ```
   build-essential
   cmake
   g++
   python3-dev
   ```

3. **Check Streamlit Cloud logs:**
   - Go to your app dashboard
   - Click "Manage app" → "Logs"
   - Look for errors during `ncrystal` or `nbragg` installation

4. **Common issues:**

   **Issue:** `packages.txt` not found
   - **Solution:** Ensure the file is in the repository root, not in a subdirectory

   **Issue:** Build timeout
   - **Solution:** NCrystal compilation can take time. If it times out, try:
     - Restarting the deployment
     - Reducing other dependencies temporarily
     - Contacting Streamlit support for increased timeout

   **Issue:** Wrong Python version
   - **Solution:** Streamlit Cloud uses Python 3.9+ by default, which is compatible with nbragg. If needed, specify version in `.python-version` file

5. **Force rebuild:**
   - Go to app dashboard
   - Click "⋮" (three dots menu)
   - Select "Reboot app"

#### If app works locally but not on Streamlit Cloud:

1. **Test locally with the same setup:**
   ```bash
   # Create fresh virtual environment
   python -m venv test_env
   source test_env/bin/activate  # On Windows: test_env\Scripts\activate

   # Install system packages (on Ubuntu/Debian)
   sudo apt-get install build-essential cmake g++ python3-dev

   # Install Python packages
   pip install -r requirements.txt

   # Run test script
   python test_nbragg_install.py

   # Run Streamlit app
   streamlit run streamlit_app.py
   ```

2. **If test fails locally:**
   - Check error messages from `test_nbragg_install.py`
   - Ensure all system dependencies are installed
   - Try: `pip install --upgrade nbragg`

3. **If test passes locally but fails on Streamlit Cloud:**
   - Check Streamlit Cloud Python version
   - Verify `packages.txt` is committed to git
   - Check deployment logs for compilation errors

## Testing Installation

### Local Testing

Run the installation test script:

```bash
python test_nbragg_install.py
```

This script will:
- ✓ Test all package imports
- ✓ Test NCrystal functionality
- ✓ Test nbragg functionality
- ✓ Test frame_overlap with Analysis import

Expected output:
```
============================================================
nbragg Installation Test Suite
============================================================

Testing package imports...
------------------------------------------------------------
✓ Streamlit                     [OK]
✓ Plotly                        [OK]
✓ NumPy                         [OK]
✓ Pandas                        [OK]
✓ SciPy                         [OK]
✓ Matplotlib                    [OK]
✓ scikit-image                  [OK]
✓ tqdm                          [OK]
✓ lmfit                         [OK]
✓ NCrystal (nbragg dependency)  [OK]
✓ nbragg                        [OK]
------------------------------------------------------------

...

============================================================
TEST SUMMARY
============================================================
✓ Package Imports               [PASS]
✓ NCrystal                      [PASS]
✓ nbragg Functionality          [PASS]
✓ frame_overlap Import          [PASS]
============================================================

✓ All tests passed! nbragg is correctly installed.
```

### Streamlit Cloud Testing

After deployment:

1. Open your app URL (e.g., `https://frame-overlap.streamlit.app`)

2. Check the sidebar - Stage 6 (Analysis nbragg) should be available without warnings

3. Test the Analysis feature:
   - Upload or use default data
   - Enable all pipeline stages (1-5)
   - Enable "Apply nbragg Analysis" in Stage 6
   - Click "Run Pipeline"
   - Verify Analysis results appear

## Files Checklist

Before deployment, ensure these files exist:

- [ ] `streamlit_app.py` (main app file)
- [ ] `requirements.txt` (Python dependencies)
- [ ] `packages.txt` (system dependencies)
- [ ] `src/frame_overlap/__init__.py` (imports Analysis without try/except)
- [ ] `.streamlit/config.toml` (optional, for theming)
- [ ] `test_nbragg_install.py` (for testing)

## Deployment URL

Once deployed, your app will be available at:
```
https://frame-overlap.streamlit.app
```

(Or your custom URL if configured)

## Further Resources

- **Streamlit deployment docs:** https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
- **packages.txt format:** https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/app-dependencies
- **NCrystal documentation:** https://github.com/mctools/ncrystal
- **nbragg documentation:** https://pypi.org/project/nbragg/

## Support

If you continue to experience issues:

1. Run `python test_nbragg_install.py` and save the output
2. Check Streamlit Cloud logs (Manage app → Logs)
3. Open an issue on GitHub with:
   - Test script output
   - Deployment log snippet
   - Error messages
   - Python version (from logs)
