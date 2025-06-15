#!/bin/bash

echo "🚂 Railway Deployment Script for Iris Classification Backend"
echo "============================================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "📦 Training models first..."
    python train_and_upload.py
fi

# Login to Railway (if not already logged in)
echo "🔐 Checking Railway authentication..."
railway whoami || railway login

# Initialize project if needed
if [ ! -f "railway.toml" ]; then
    echo "🚀 Initializing Railway project..."
    railway init
fi

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "📡 Your API will be available at the Railway-provided URL"
echo "🔗 Check your Railway dashboard for the exact URL"
echo "⚙️  Don't forget to update PYTHON_SERVICE_URL in your Vercel environment variables"
