# Deploying to Streamlit Cloud

This guide explains how to deploy the Frame Overlap Interactive Explorer to Streamlit Community Cloud.

## Prerequisites

1. A GitHub account
2. The frame_overlap repository pushed to GitHub
3. A Streamlit Community Cloud account (free at https://streamlit.io/cloud)

## Deployment Steps

### 1. Prepare the Repository

Make sure the following files are in your repository:
- `streamlit_app.py` - The main Streamlit application
- `requirements.txt` - Python dependencies for Streamlit Cloud
- `.streamlit/config.toml` - Streamlit configuration
- `notebooks/iron_powder.csv` - Sample data
- `notebooks/openbeam.csv` - Sample data
- `src/frame_overlap/` - The frame_overlap package source

### 2. Sign Up for Streamlit Community Cloud

1. Go to https://streamlit.io/cloud
2. Sign up using your GitHub account
3. Authorize Streamlit to access your GitHub repositories

### 3. Deploy the App

1. Click "New app" in your Streamlit Cloud dashboard
2. Select your repository: `TsvikiHirsh/frame_overlap`
3. Set the branch: `main` (or your current branch)
4. Set the main file path: `streamlit_app.py`
5. Click "Deploy!"

The app will be deployed at: `https://frame-overlap.streamlit.app`

### 4. Update the App

Once deployed, Streamlit Cloud will automatically update the app whenever you push changes to the specified branch on GitHub.

## App URL

After deployment, your app will be available at:
- **https://frame-overlap.streamlit.app**

This URL is already included in the README.md badges and links.

## Troubleshooting

### App Won't Start

- Check the logs in Streamlit Cloud dashboard
- Verify all required files are in the repository
- Make sure `requirements.txt` includes all dependencies

### Import Errors

If you see import errors, make sure:
- The `src/` directory is in the repository
- The `sys.path.insert(0, 'src')` line is in `streamlit_app.py`

### Data Files Not Found

If data files aren't loading:
- Verify `notebooks/iron_powder.csv` and `notebooks/openbeam.csv` exist in the repo
- Check the file paths in `streamlit_app.py`

### Plotting Issues

If plots don't render:
- Check that matplotlib backend is set to 'Agg'
- Verify plotly is in requirements.txt
- Review the matplotlib to plotly conversion function

## Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the app
streamlit run streamlit_app.py
```

The app will open in your browser at http://localhost:8501

## Configuration

The `.streamlit/config.toml` file controls:
- Theme colors
- Server settings
- Browser behavior

Modify this file to customize the appearance and behavior of your app.

## Monitoring

Streamlit Cloud provides:
- Real-time logs
- Resource usage monitoring
- App analytics (if enabled)

Access these from your Streamlit Cloud dashboard.
