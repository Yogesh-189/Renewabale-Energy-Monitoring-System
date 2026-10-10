# Deploying to the cloud

After deployment, **one web address serves everything**: the dashboard at `/`, the API docs at `/docs`.
The API also gets real internet access from the cloud server, so the live Open-Meteo forecast works there
even when your college Wi-Fi blocks it.

## Step 0 - make sure the needed files are in GitHub
The server builds from your repository, so these must be committed (some `.gitignore` files exclude them):

```bash
git ls-files | findstr /i "joblib Plant_1 dashboard src"      # Windows PowerShell
```
You should see `models/forecast_model.joblib`, `data/Plant_1_Generation_Data.csv`,
`data/Plant_1_Weather_Sensor_Data.csv`, `dashboard/index.html` and the files in `src/`.
If the model is missing the server trains it on first start (about a minute) - but the two CSV files are required.
If `git` says a file is ignored, remove that line from `.gitignore`, then `git add -f <file>`.

## Option A - Render (free, no Docker needed)
1. Push the latest code: `git add .` -> `git commit -m "Add deployment files"` -> `git push`
2. Sign in at https://render.com with your GitHub account.
3. **New +** -> **Blueprint** -> pick your repository. Render reads `render.yaml` and creates the service.
   (Or **New +** -> **Web Service** and enter: build `pip install -r requirements.txt`,
   start `cd src && uvicorn api:app --host 0.0.0.0 --port $PORT`.)
4. Wait for the first build (a few minutes). Render gives you an address like `https://renewable-energy-monitoring.onrender.com`.
5. Open it: the dashboard loads. Also try `/docs`, `/health` and `/weather/diagnose`.

Free-tier behaviour to know before a demo: the service **sleeps after ~15 minutes without visitors** and the next
request takes 30-60 seconds to wake it. Open the link a minute before presenting.

## Option B - Docker (any cloud, or your own laptop)
```bash
docker build -t solar-monitoring .
docker run -p 8000:8000 solar-monitoring
```
Then open http://localhost:8000. The same image runs on Google Cloud Run, Azure Container Apps, AWS App Runner or Railway.

## Hosting the dashboard separately (optional)
If the dashboard is hosted on a different address than the API (for example GitHub Pages), open it with the API
address in the URL: `https://your-dashboard-page/?api=https://your-api-address`

## Known limits of this deployment (be honest about them in a review)
- The data is the fixed Kaggle file and the sensor feed is a simulator - deploying does not make them real-time hardware data.
- The API has **no login yet**: anyone with the link can read it. Authentication is the next planned step.
- The free tier has 512 MB RAM; this app uses well under that, but a very large dataset would not fit.
