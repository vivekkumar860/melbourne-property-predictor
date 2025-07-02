#!/usr/bin/env python3
"""
Test script to identify and fix bugs in the Melbourne Property Price Prediction app
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    try:
        df = pd.read_csv('melb_data.csv')
        print(f"✓ Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # Check required columns
        required_cols = ['Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 
                        'Landsize', 'BuildingArea', 'YearBuilt', 'Price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return False
        else:
            print("✓ All required columns present")
            return True
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\nTesting model loading...")
    try:
        if os.path.exists('price_prediction_model.pkl') and os.path.exists('preprocessor.pkl'):
            with open('price_prediction_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            print("✓ Model and preprocessor loaded successfully")
            return model, preprocessor
        else:
            print("✗ Model files not found")
            return None, None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None

def test_prediction(model, preprocessor):
    """Test prediction functionality"""
    print("\nTesting prediction...")
    try:
        # Create test input
        test_data = pd.DataFrame({
            'Rooms': [3],
            'Type': ['h'],
            'Distance': [10.0],
            'Bedroom2': [3],
            'Bathroom': [2],
            'Car': [2],
            'Landsize': [500],
            'BuildingArea': [150],
            'YearBuilt': [2000]
        })
        
        # Preprocess and predict
        input_processed = preprocessor.transform(test_data)
        prediction = model.predict(input_processed)[0]
        
        print(f"✓ Prediction successful: ${prediction:,.0f}")
        return True
    except Exception as e:
        print(f"✗ Error making prediction: {e}")
        return False

def main():
    print("=== Melbourne Property Price Prediction App - Bug Test ===\n")
    
    # Test 1: Data loading
    data_ok = test_data_loading()
    
    # Test 2: Model loading
    model, preprocessor = test_model_loading()
    
    # Test 3: Prediction
    if model and preprocessor:
        prediction_ok = test_prediction(model, preprocessor)
    else:
        prediction_ok = False
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Data Loading: {'✓ PASS' if data_ok else '✗ FAIL'}")
    print(f"Model Loading: {'✓ PASS' if model else '✗ FAIL'}")
    print(f"Prediction: {'✓ PASS' if prediction_ok else '✗ FAIL'}")
    
    if data_ok and model and prediction_ok:
        print("\n🎉 All tests passed! The app should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main() 