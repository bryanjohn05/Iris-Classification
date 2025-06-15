# 🚂 Railway Backend for Iris Classification

This is the Python backend service that provides real machine learning model predictions for the Iris Classification frontend.

## 🚀 Quick Deploy to Railway

### Option 1: Deploy from GitHub (Recommended)

1. **Push this folder to a GitHub repository**
2. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Select the `railway_backend` folder as root

3. **Railway will automatically**:
   - Detect the Dockerfile
   - Build and deploy your service
   - Provide a public URL

### Option 2: Deploy with Railway CLI

1. **Install Railway CLI**:
   \`\`\`bash
   npm install -g @railway/cli
   \`\`\`

2. **Login and deploy**:
   \`\`\`bash
   railway login
   railway init
   railway up
   \`\`\`

## 📦 What's Included

- **FastAPI Application**: Modern Python web framework
- **ML Models**: All 5 iris classification models
- **Docker Configuration**: Ready for containerized deployment
- **Health Checks**: Monitoring endpoints
- **CORS Support**: Works with any frontend domain

## 🔧 Local Development

1. **Install dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Train models** (if not already done):
   \`\`\`bash
   python train_and_upload.py
   \`\`\`

3. **Run the server**:
   \`\`\`bash
   python main.py
   \`\`\`

4. **Test the API**:
   - Open http://localhost:8000
   - View docs at http://localhost:8000/docs

## 📡 API Endpoints

- `GET /` - Service information
- `GET /health` - Health check
- `POST /predict` - Single model prediction
- `POST /predict-all` - All models prediction
- `GET /model-performance` - Performance metrics

## 🌐 After Deployment

1. **Get your Railway URL** (e.g., `https://your-app.railway.app`)
2. **Update your Vercel environment**:
   - Go to Vercel dashboard
   - Add environment variable: `PYTHON_SERVICE_URL=https://your-app.railway.app`
   - Redeploy your frontend

## 🔍 Monitoring

- **Railway Dashboard**: Monitor logs, metrics, and deployments
- **Health Endpoint**: Check `/health` for service status
- **API Documentation**: Available at `/docs`

## 🛠️ Troubleshooting

### Models Not Loading
- Ensure `models/` directory contains all `.joblib` files
- Check Railway logs for loading errors
- Verify file paths in the application

### CORS Issues
- Update `allow_origins` in `main.py` with your frontend domain
- For development, `["*"]` allows all origins

### Performance Issues
- Railway provides 512MB RAM by default
- Upgrade plan if needed for larger models
- Monitor memory usage in Railway dashboard

## 📊 Model Files Required

\`\`\`
models/
├── scaler.joblib
├── RandomForest_model.joblib
├── SVM_model.joblib
├── KNN_model.joblib
├── DecisionTree_model.joblib
└── LogisticRegression_model.joblib
\`\`\`

## 🎯 Production Tips

1. **Environment Variables**: Use Railway's environment variables for sensitive config
2. **Monitoring**: Set up Railway's monitoring and alerts
3. **Scaling**: Railway auto-scales based on traffic
4. **Updates**: Push to GitHub to trigger automatic redeployments
