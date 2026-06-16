# Campus Placement Prediction — Deployment

Quick steps to deploy this FastAPI app to Railway (or similar PaaS):

1. Ensure the repository contains `placement.csv`, `index.html`, `app.js`, and `styles.css` in the project root.

2. Confirm `requirements.txt` is present (this repo includes it).

3. Push the repository to GitHub and connect it to Railway:
   - Create a new project on Railway and link the GitHub repo.
   - Railway will detect Python projects; set the Build/Install command to:

```
pip install -r requirements.txt
```

4. Set the Start/Run command (or use the provided `Procfile`):

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

5. Environment & notes:
   - The app requires `placement.csv` at runtime; add it to the repo or use Railway storage.
   - The API serves static frontend files from the repo root (`index.html`, `app.js`, `styles.css`).
   - If startup training is slow, consider precomputing models and loading pickled models instead.

Optional: use the included `Dockerfile` to build a container image instead of the platform buildpacks.

If you want, I can create a pre-trained model file and update the app to load it at startup to speed deployments.
