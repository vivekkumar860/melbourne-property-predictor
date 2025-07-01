# Melbourne Property Price Prediction

A comprehensive Streamlit web application that predicts property prices in Melbourne, Australia using machine learning, computer vision, and geospatial analysis.

## 🏠 Features

- **ML-Powered Predictions**: Advanced machine learning model for accurate price predictions
- **Computer Vision**: ResNet50 integration for property image analysis
- **Interactive Maps**: Folium-based maps with Points of Interest (POI) analysis
- **Data Visualization**: Comprehensive charts and insights
- **Google Maps Integration**: Real-time POI extraction and analysis
- **Responsive UI**: Modern, user-friendly interface

## 🚀 Quick Deploy Options

### Option 1: Streamlit Cloud (Recommended - Free)
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Select this repository: `vivekkumar860/melbourne-property-predictor`
4. Set main file path: `app.py`
5. Click "Deploy!"

### Option 2: Railway (Free tier)
1. Visit [railway.app](https://railway.app/)
2. Connect your GitHub account
3. Select this repository
4. Railway will auto-detect and deploy

### Option 3: Render (Free tier)
1. Go to [render.com](https://render.com/)
2. Connect GitHub and select this repository
3. Create Web Service
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### Option 4: Heroku
1. Install Heroku CLI
2. Run: `heroku create your-app-name`
3. Run: `git push heroku main`

## 🛠️ Local Development

### Prerequisites
- Python 3.9+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/vivekkumar860/melbourne-property-predictor.git
cd melbourne-property-predictor

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t property-predictor .
docker run -p 8501:8501 property-predictor

# Or use Docker Compose
docker-compose up
```

## 📊 Dataset

- **Source**: Melbourne housing market dataset
- **Size**: 12,968 properties
- **Price Range**: $85,000 - $2,350,000
- **Features**: Property type, rooms, bathrooms, land size, location, etc.

## 🧠 Model Architecture

- **Primary Model**: Random Forest Regressor
- **Computer Vision**: ResNet50 for image analysis
- **Preprocessing**: StandardScaler, OneHotEncoder
- **Evaluation Metrics**: RMSE, MAE, R²

## 🔧 Configuration

### Environment Variables
- `GOOGLE_MAPS_API_KEY`: For POI analysis (optional)
- `PYTHONUNBUFFERED=1`: For Docker deployment

### API Keys Required
- Google Maps API (optional, for enhanced POI features)

## 📱 Usage

1. **Input Property Details**: Fill in property attributes
2. **Upload Images** (optional): Add property photos for visual analysis
3. **Get Predictions**: Receive price estimates with confidence intervals
4. **Explore Insights**: View market analysis and trends
5. **Interactive Maps**: Explore locations and nearby amenities

## 🎯 Key Features

- **Price Prediction**: ML-based property valuation
- **Image Analysis**: Visual feature extraction
- **Geospatial Analysis**: Location-based insights
- **Market Insights**: Data visualization and trends
- **POI Integration**: Nearby amenities analysis
- **Responsive Design**: Mobile-friendly interface

## 📈 Performance

- **Model Accuracy**: High prediction accuracy with cross-validation
- **Response Time**: Fast inference with caching
- **Scalability**: Optimized for production deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Melbourne housing dataset
- Streamlit framework
- TensorFlow/Keras
- Google Maps API
- Open source community

---

**Live Demo**: [Deploy to see it in action!](https://share.streamlit.io/)

**Repository**: https://github.com/vivekkumar860/melbourne-property-predictor 