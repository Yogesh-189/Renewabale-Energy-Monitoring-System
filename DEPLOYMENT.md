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

## If the live weather says "429 Too Many Requests"

Open-Meteo limits free use per outgoing address, and a free Render server shares its address with many other apps.
The app now (1) keeps each place's forecast for 20 minutes instead of asking on every 30-second refresh, (2) never retries
a 429 and waits before asking again, and (3) when the server cannot get the weather, the **viewer's browser fetches it itself**
and sends it to `POST /forecast/live-weather/from-browser`, so the Satellite Weather tab keeps working. The tab tells you which
route was used. `WEATHER_CACHE_MINUTES` (default 20) changes the cache time.

## Security (login)

With the login on, the dashboard shows a sign-in box and every API endpoint answers **401 Unauthorized** to anyone
without a valid token. Only `/health` (needed by Render), `/auth/status`, `/auth/login` and the dashboard page itself stay public.
HTTPS is provided by Render automatically; the API also sends HSTS and other browser-safety headers over HTTPS.

**Turn it on in Render:** your service -> **Environment** -> **Add Environment Variable**:

| Key | Value |
|---|---|
| `AUTH_PASSWORD` | a long passphrase (10+ characters). This switches the login on. |
| `AUTH_SECRET` | a long random string. Create one with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `AUTH_USERNAME` | optional, default is `admin` |
| `AUTH_TOKEN_HOURS` | optional, how long a sign-in lasts, default 8 |

Click **Save Changes**; Render redeploys. Without `AUTH_PASSWORD` the API stays open (handy on your laptop, wrong for a public server).

**Try it on your laptop first** (PowerShell, in the project folder):
```
$env:AUTH_PASSWORD = "choose-a-long-passphrase"
$env:AUTH_SECRET = "paste-a-long-random-string"
cd src
uvicorn api:app --reload --port 8000
```
Open http://localhost:8000: you get the sign-in box. To go back to open mode, close that terminal and open a new one.

**Show it to a reviewer:** open `/docs`, run `GET /forecast` -> **401**. Run `POST /auth/login` with the username and password,
copy `access_token`, click **Authorize**, paste it, and `GET /forecast` now returns data.

**Limits (say them honestly):** one shared account, no user database or roles; a token cannot be cancelled before it expires
(Sign out only forgets it in the browser); the password is an environment variable, not a hashed database entry.

## Known limits of this deployment (be honest about them in a review)
- The data is the fixed Kaggle file and the sensor feed is a simulator - deploying does not make them real-time hardware data.
- Login is only active once `AUTH_PASSWORD` is set (see Security above).
- The free tier has 512 MB RAM; this app uses well under that, but a very large dataset would not fit.
