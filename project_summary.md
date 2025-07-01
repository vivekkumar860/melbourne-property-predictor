# Melbourne Property Price Prediction - Project Summary

## Project Overview
The Melbourne Property Price Prediction application is a Streamlit-based web application designed to accurately predict property prices in Melbourne, Australia. The application combines traditional machine learning techniques with advanced computer vision and geospatial analysis to provide users with data-driven property valuations and insights.

## Dataset Information
- **Source**: Melbourne housing market dataset
- **Size**: 12,968 properties
- **Price Range**: $85,000 - $2,350,000
- **Average Price**: $982,626
- **Key Features**: Property type, rooms, bathrooms, car spaces, land size, building area, location (suburb), year built
- **Data Cleaning**: Handled missing values, outliers, and inconsistent entries
- **Data Split**: 80% training, 20% testing with stratification by suburb

## Machine Learning Implementation
- **Primary Model**: Random Forest Regressor
  - n_estimators: 200
  - max_depth: 25
  - min_samples_split: 5
  - min_samples_leaf: 2
- **Model Performance**:
  - RMSE: $84,627
  - MAE: $56,912
  - R²: 0.87
- **Feature Importance**: Suburb, Land Size, Rooms, Building Area, and Property Type were the most significant predictors
- **Prediction Pipeline**:
  - Data validation and preprocessing of user inputs
  - Model inference to generate price predictions
  - Confidence interval generation (±10%)
  - Price breakdown into land value and building value components

## User Interface Components
- **Main Prediction Interface**:
  - Input fields for all property attributes
  - Suburb selection with location mapping
  - Interactive map visualization
  - Prediction results with price gauge
  - Component value breakdown

- **Data Insights Section**:
  - Property price distribution
  - Property type comparison
  - Suburb price analysis
  - Feature importance visualization
  - Correlation analysis

- **POI Explorer**:
  - Nearby amenities visualization
  - Schools, hospitals, transit stations, and supermarkets
  - Distance-based scoring system

- **Guide & FAQ Section**:
  - Usage instructions
  - Methodology explanation
  - Result interpretation guide
  - Common questions and answers

## Advanced Features

### Google Maps Integration
- Real-time Points of Interest (POI) extraction using Google Maps Places API
- Analysis of nearby amenities (schools, hospitals, supermarkets, transit stations)
- Interactive map with POI markers color-coded by type
- Distance-based amenity scoring system

### ResNet50 Visual Analysis
- Property image analysis using ResNet50 deep learning model
- Extraction of visual features from property images
- Detection of value-adding features (pools, renovations, good views)
- Visual quality scoring integrated into price prediction

### Data Visualization
- Dynamic price gauge showing predicted value
- Feature importance plots
- Property distribution maps
- Suburb comparison charts
- Building vs. land value breakdown

## Technical Implementation

### Project Structure
- Modular code organization with separate components for:
  - Data preprocessing and cleaning
  - Model training and inference
  - API integrations
  - Visual analysis
  - UI components

### Key Dependencies
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning implementation
- **TensorFlow/Keras**: Deep learning for image analysis
- **Folium**: Interactive maps
- **Matplotlib/Plotly**: Data visualization
- **Google Maps API**: Geospatial analysis

### Performance Optimizations
- Resource-intensive operations cached using `@st.cache_resource`
- Efficient data loading with preprocessing
- Session state management for persistent user experience

## Bug Fixes and Improvements

### Critical Fixes
- **Negative Building Value Issue**:
  - Fixed calculation to ensure balanced distribution between land and building values
  - Added validation to prevent negative component values
  - Implemented ratio capping between 20-80%

- **App Reset Prevention**:
  - Implemented session state management to maintain prediction results
  - Added state persistence for user inputs
  - Created keyed components to prevent unwanted reruns

### UI/UX Enhancements
- Background image and styling improvements
- Responsive layout design
- Custom icons for property types
- Progress indicators for long-running operations
- Tooltips and help text for user guidance

## Deployment Information

### Local Setup
1. Clone the repository
2. Install dependencies via `pip install -r requirements.txt`
3. Set up API keys for Google Maps integration
4. Run the application using `streamlit run app.py`

### System Requirements
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- Internet connection for API access

## Future Development Roadmap

### Planned Features
- User accounts and saved property comparisons
- Historical price trends and forecasting
- Mobile-friendly responsive design
- Export functionality for reports

### Technical Improvements
- Migration to more efficient model architecture
- A/B testing for UI enhancements
- Database integration
- API development for third-party integrations

## Acknowledgments
- Melbourne housing market dataset
- Streamlit framework
- TensorFlow and scikit-learn libraries
- Google Maps API for geospatial analysis 