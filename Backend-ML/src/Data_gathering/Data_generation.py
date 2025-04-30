#!/usr/bin/env python3
"""
Weather Impact Data Generator with Balanced Travel Disruption Classes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import logging
import argparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataGenerator:
    """Generates synthetic weather data with balanced travel disruption classes"""
    
    def __init__(self, seed: int = 42):
        """Initialize with optional seed for reproducibility"""
        self.seed = seed
        self._initialize_random_generators()
        
    def _initialize_random_generators(self) -> None:
        """Initialize all random number generators with the specified seed"""
        np.random.seed(self.seed)
        random.seed(self.seed)
        logger.debug(f"Initialized random generators with seed {self.seed}")
    
    def _generate_weather_features(self, date: datetime) -> Dict[str, Any]:
        """Generate weather features with more extreme weather events"""
        # Adjusted probabilities to create more impactful weather
        is_rainy = np.random.choice([True, False], p=[0.5, 0.5])  # 50% chance of rain
        is_stormy = np.random.choice([True, False], p=[0.3, 0.7]) if is_rainy else False  # 30% chance of storm when rainy
        
        # Temperature with seasonal pattern
        day_of_year = date.timetuple().tm_yday
        temperature = 10 + 15 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5)  # More variation
        
        # Rainfall - higher values more likely
        rainfall = np.random.gamma(shape=2, scale=3) if is_rainy else 0
        
        # Wind speed - higher base values
        wind_speed = np.random.weibull(2) * (25 if is_stormy else 10)
        
        # Other weather features
        humidity = np.random.uniform(70, 100) if is_rainy else np.random.uniform(30, 70)
        pressure = np.random.normal(1010, 10) - (25 if is_stormy else 0)  # More variation
        visibility = max(1, np.random.normal(10, 5) - (rainfall * 0.8))  # Lower visibility
        
        # Weather condition classification
        if is_stormy:
            weather_condition = "storm"
        elif is_rainy:
            weather_condition = "rain"
        else:
            weather_condition = np.random.choice(["sunny", "cloudy", "fog"], p=[0.4, 0.5, 0.1])  # More cloudy days
        
        return {
            "temperature": temperature,
            "rainfall": rainfall,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "pressure": pressure,
            "visibility": visibility,
            "weather_condition": weather_condition,
            "is_stormy": is_stormy
        }
    
    def _determine_severity(self, wind_speed: float, rainfall: float) -> str:
        """More sensitive severity classification"""
        if wind_speed > 20 or rainfall > 10:  # Lower thresholds
            return "severe"
        if wind_speed > 10 or rainfall > 3:
            return "moderate"
        return "mild"
    
    def _determine_travel_impact(self, severity: str) -> str:
        """Balanced travel disruption probabilities"""
        if severity == "severe":
            return np.random.choice(["delay", "cancellation"], p=[0.4, 0.6])  # More cancellations
        if severity == "moderate":
            return np.random.choice(["none", "delay", "cancellation"], p=[0.3, 0.6, 0.1])  # Some cancellations
        return np.random.choice(["none", "delay"], p=[0.7, 0.3])  # More mild delays
    
    def _determine_road_condition(self, weather_condition: str) -> str:
        """Determine road condition based on weather"""
        conditions = {
            "rain": "wet",
            "storm": "flooded",
            "fog": "slippery",
            "cloudy": "dry"
        }
        return conditions.get(weather_condition, "dry")
    
    def generate_dataset(self, num_rows: int, start_date: str = "2023-01-01") -> pd.DataFrame:
        """
        Generate a synthetic weather dataset with balanced travel disruption classes
        
        Args:
            num_rows: Number of rows/days to generate
            start_date: Starting date for the dataset (YYYY-MM-DD format)
            
        Returns:
            pandas.DataFrame: Generated weather dataset
        """
        data = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        for _ in range(num_rows):
            weather = self._generate_weather_features(current_date)
            severity = self._determine_severity(weather["wind_speed"], weather["rainfall"])
            travel_disruption = self._determine_travel_impact(severity)
            road_condition = self._determine_road_condition(weather["weather_condition"])
            
            traffic_density = np.random.poisson(50 if current_date.weekday() < 5 else 30)
            
            record = {
                "date": current_date.strftime("%Y-%m-%d"),
                "temperature": round(weather["temperature"], 1),
                "rainfall": round(weather["rainfall"], 1),
                "humidity": round(weather["humidity"], 1),
                "wind_speed": round(weather["wind_speed"], 1),
                "pressure": round(weather["pressure"], 1),
                "visibility": round(weather["visibility"], 1),
                "weather_condition": weather["weather_condition"],
                "severity": severity,
                "travel_disruption": travel_disruption,
                "road_condition": road_condition,
                "traffic_density": traffic_density
            }
            
            data.append(record)
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(data)
        
        # Ensure minimum counts for each class
        min_count = num_rows // 10  # At least 10% for minority classes
        disruption_counts = df['travel_disruption'].value_counts()
        
        if disruption_counts.get('cancellation', 0) < min_count:
            # Find severe weather days and force some cancellations
            severe_days = df[df['severity'] == 'severe']
            if len(severe_days) > 0:
                n_needed = min_count - disruption_counts.get('cancellation', 0)
                to_change = severe_days.sample(min(n_needed, len(severe_days)))
                df.loc[to_change.index, 'travel_disruption'] = 'cancellation'
        
        if disruption_counts.get('delay', 0) < min_count:
            # Find moderate weather days and force some delays
            moderate_days = df[df['severity'] == 'moderate']
            if len(moderate_days) > 0:
                n_needed = min_count - disruption_counts.get('delay', 0)
                to_change = moderate_days.sample(min(n_needed, len(moderate_days)))
                df.loc[to_change.index, 'travel_disruption'] = 'delay'
        
        return df

def main():
    """Command-line interface for data generation"""
    parser = argparse.ArgumentParser(description="Generate synthetic weather impact data")
    parser.add_argument("-n", "--num_rows", type=int, default=1200,
                       help="Number of rows/days to generate")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "weather_data1.csv")

    parser.add_argument("-o", "--output", 
                    default=default_output,
                    help="Output file path")
    parser.add_argument("-s", "--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--start_date", default="2023-01-01",
                       help="Start date in YYYY-MM-DD format")
    
    args = parser.parse_args()
    
    logger.info(f"Generating {args.num_rows} rows of weather data starting from {args.start_date}")
    
    try:
        generator = WeatherDataGenerator(seed=args.seed)
        df = generator.generate_dataset(args.num_rows, args.start_date)
        
        # Print class distribution
        logger.info("\nTravel Disruption Class Distribution:")
        logger.info(df['travel_disruption'].value_counts())
        
        logger.info("\nSeverity Class Distribution:")
        logger.info(df['severity'].value_counts())
        
        df.to_csv(args.output, index=False)
        logger.info(f"\nSuccessfully generated data. Saved to {args.output}")
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()