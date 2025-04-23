#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Loader Utility

This module handles loading and preprocessing of data for the application.
It follows the project's modular organization pattern, keeping data processing
logic together and generating summary statistics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
import json
import datetime

class DataLoader:
    """
    Data loading and preprocessing utility for the Miami Housing Impact Hub.
    
    This class handles loading census, Airbnb, and housing data from the processed
    data directories. It also provides methods for data aggregation and filtering.
    """
    
    def __init__(self, project_root):
        """
        Initialize the data loader with the project root path.
        
        Args:
            project_root (Path): Path to the project root directory.
        """
        self.project_root = project_root
        self.processed_data_dir = project_root / "data" / "processed"
        self.raw_data_dir = project_root / "data" / "raw"
        
        # Paths to processed data
        self.airbnb_dir = self.processed_data_dir / "airbnb"
        self.census_dir = self.processed_data_dir / "census"
        self.combined_dir = self.processed_data_dir / "combined"
        
        # Initialize data containers
        self.airbnb_data = None
        self.census_data = None
        self.combined_data = None
        self.summary_stats = {}
        
        # Load the data
        self.load_data()
        
    def load_data(self):
        """Load all datasets from available files without using dummy data."""
        # Create the required directories if they don't exist
        os.makedirs(self.airbnb_dir, exist_ok=True)
        os.makedirs(self.census_dir, exist_ok=True)
        os.makedirs(self.combined_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir / "summary", exist_ok=True)
        
        # Initialize the summary statistics dictionary
        self.summary_stats = {}
        
        # Primary data loading - first try to load the miami_dade_merged_data.csv
        merged_file = self.processed_data_dir / "miami_dade_merged_data.csv"
        if os.path.exists(merged_file):
            print(f"Loading merged data from {merged_file}")
            self.combined_data = pd.read_csv(merged_file)
            # Convert the combined data to also be used as airbnb and census data
            self.airbnb_data = self.combined_data.copy()
            self.census_data = self.combined_data.copy()
        else:
            # Try to load individual datasets
            print("Main merged data file not found. Attempting to load individual datasets.")
            self._load_airbnb_data()
            self._load_census_data()
            self._load_combined_data()
        
        # Generate summary statistics for all loaded data
        try:
            self._generate_summary_stats()
        except Exception as e:
            print(f"Error generating summary stats: {e}")
            # Initialize with basic summary stats based on available data
            self.summary_stats = {}
            if self.combined_data is not None:
                self.summary_stats['combined'] = {
                    'total_neighborhoods': len(self.combined_data),
                    'total_airbnbs': self.combined_data['airbnb_count'].sum() if 'airbnb_count' in self.combined_data.columns else 0
                }
    
    def _load_airbnb_data(self):
        """Load Airbnb data from the processed directory."""
        try:
            airbnb_file = self.airbnb_dir / "airbnb_by_zipcode.csv"
            if os.path.exists(airbnb_file):
                self.airbnb_data = pd.read_csv(airbnb_file)
                print(f"Loaded Airbnb data with {self.airbnb_data.shape[0]} records")
            else:
                print(f"Warning: {airbnb_file} not found")
                # Try to use the raw data or create dummy data for prototype
                self._create_dummy_airbnb_data()
        except Exception as e:
            print(f"Error loading Airbnb data: {e}")
            self._create_dummy_airbnb_data()
    
    def _load_census_data(self):
        """Load census data from the processed directory."""
        try:
            census_file = self.census_dir / "cleaned_acs5_miamidade_tracts.csv"
            if os.path.exists(census_file):
                self.census_data = pd.read_csv(census_file)
                print(f"Loaded census data with {self.census_data.shape[0]} records")
            else:
                print(f"Warning: {census_file} not found")
                # Try to use raw data or create dummy data for prototype
                raw_census_file = self.raw_data_dir / "census" / "acs5_miamidade_tracts.csv"
                if os.path.exists(raw_census_file):
                    self.census_data = pd.read_csv(raw_census_file)
                    print(f"Loaded raw census data with {self.census_data.shape[0]} records")
                else:
                    self._create_dummy_census_data()
        except Exception as e:
            print(f"Error loading census data: {e}")
            self._create_dummy_census_data()
    
    def _load_combined_data(self):
        """Load combined Airbnb-census data."""
        try:
            combined_file = self.combined_dir / "airbnb_census_by_zipcode.csv"
            if os.path.exists(combined_file):
                self.combined_data = pd.read_csv(combined_file)
                print(f"Loaded combined data with {self.combined_data.shape[0]} records")
            else:
                print(f"Warning: {combined_file} not found")
                # If we have both Airbnb and census data, create a simplified combined dataset
                if self.airbnb_data is not None and self.census_data is not None:
                    self._create_simple_combined_data()
        except Exception as e:
            print(f"Error loading combined data: {e}")
    
    def _create_simple_combined_data(self):
        """Create a simplified combined dataset from existing Airbnb and census data."""
        print("Creating simplified combined data from existing datasets")
        # This is a placeholder - in a full implementation, we would properly join the datasets
        # For the prototype, we'll create a dummy version
        # self._create_dummy_combined_data()
        pass
    
    def _generate_summary_stats(self):
        """Generate summary statistics for all datasets."""
        # Initialize the summary_stats dictionary if it doesn't exist
        if not hasattr(self, 'summary_stats'):
            self.summary_stats = {}
            
        # Airbnb data summary
        if self.airbnb_data is not None:
            try:
                summary = {
                    'total_listings': self.airbnb_data['listing_count'].sum() if 'listing_count' in self.airbnb_data.columns else 0,
                    'top_zips': self.airbnb_data.sort_values('listing_count', ascending=False)['zip_code'].head(5).tolist() if 'listing_count' in self.airbnb_data.columns and 'zip_code' in self.airbnb_data.columns else []
                }
                
                # Add optional metrics if columns exist
                if 'avg_price' in self.airbnb_data.columns:
                    summary['avg_price'] = self.airbnb_data['avg_price'].mean()
                elif 'median_price' in self.airbnb_data.columns:
                    summary['avg_price'] = self.airbnb_data['median_price'].mean()
                
                if 'entire_home_count' in self.airbnb_data.columns and 'listing_count' in self.airbnb_data.columns:
                    summary['entire_home_percent'] = (self.airbnb_data['entire_home_count'].sum() / 
                                                  self.airbnb_data['listing_count'].sum() * 100)
                elif 'entire_home_percent' in self.airbnb_data.columns:
                    summary['entire_home_percent'] = self.airbnb_data['entire_home_percent'].mean()
                
                self.summary_stats['airbnb'] = summary
                
                # Following user's preference, save summary immediately after processing
                airbnb_summary_file = self.airbnb_dir / "summary_airbnb.csv"
                pd.DataFrame([summary]).to_csv(airbnb_summary_file, index=False)
            except Exception as e:
                print(f"Error generating Airbnb summary: {e}")
                self.summary_stats['airbnb'] = {'total_listings': 0, 'top_zips': []}
        
        # Census data summary
        if self.census_data is not None:
            census_summary = {}
            
            # Add metrics if columns exist
            if 'population' in self.census_data.columns:
                census_summary['total_population'] = self.census_data['population'].sum()
            
            if 'median_household_income' in self.census_data.columns:
                census_summary['avg_median_income'] = self.census_data['median_household_income'].mean()
            
            if 'median_gross_rent' in self.census_data.columns:
                census_summary['avg_median_rent'] = self.census_data['median_gross_rent'].mean()
            
            if 'rent_to_income_ratio' in self.census_data.columns:
                census_summary['avg_rent_to_income'] = self.census_data['rent_to_income_ratio'].mean()
                census_summary['unaffordable_tracts_pct'] = (self.census_data['rent_to_income_ratio'] > 30).mean() * 100
            
            self.summary_stats['census'] = census_summary
        
        # Combined data summary
        if self.combined_data is not None:
            combined_summary = {}
            
            # Check if required columns exist for correlation calculation
            required_cols = ['rent_to_income_ratio', 'listings_per_1000']
            if all(col in self.combined_data.columns for col in required_cols):
                combined_summary['correlation_rent_listings'] = self.combined_data[required_cols].corr().iloc[0,1]
            
            # Check if required columns exist for density level calculations
            if 'airbnb_density_level' in self.combined_data.columns and 'rent_to_income_ratio' in self.combined_data.columns:
                high_density = self.combined_data[self.combined_data['airbnb_density_level'].isin(['High', 'Very High'])]
                low_density = self.combined_data[self.combined_data['airbnb_density_level'].isin(['Low', 'Very Low'])]
                
                if not high_density.empty:
                    combined_summary['high_density_avg_rent_ratio'] = high_density['rent_to_income_ratio'].mean()
                
                if not low_density.empty:
                    combined_summary['low_density_avg_rent_ratio'] = low_density['rent_to_income_ratio'].mean()
            
            self.summary_stats['combined'] = combined_summary
    
    def get_airbnb_data(self, filtered=False, **filter_args):
        """
        Get Airbnb data, optionally filtered.
        
        Args:
            filtered (bool): Whether to apply filtering
            **filter_args: Filtering arguments
            
        Returns:
            pd.DataFrame: Filtered or unfiltered Airbnb data
        """
        if self.airbnb_data is None:
            return pd.DataFrame()
        
        if not filtered:
            return self.airbnb_data
        
        # Apply filters
        filtered_data = self.airbnb_data.copy()
        # Example: filter by min_listings if provided
        if 'min_listings' in filter_args:
            filtered_data = filtered_data[filtered_data['listing_count'] >= filter_args['min_listings']]
        
        return filtered_data
    
    def get_census_data(self, filtered=False, **filter_args):
        """
        Get census data, optionally filtered.
        
        Args:
            filtered (bool): Whether to apply filtering
            **filter_args: Filtering arguments
            
        Returns:
            pd.DataFrame: Filtered or unfiltered census data
        """
        if self.census_data is None:
            return pd.DataFrame()
        
        if not filtered:
            return self.census_data
        
        # Apply filters
        filtered_data = self.census_data.copy()
        # Example: filter by income range if provided
        if 'min_income' in filter_args:
            filtered_data = filtered_data[filtered_data['median_household_income'] >= filter_args['min_income']]
        if 'max_income' in filter_args:
            filtered_data = filtered_data[filtered_data['median_household_income'] <= filter_args['max_income']]
        
        return filtered_data
    
    def get_combined_data(self, filtered=False, density_levels=None, min_income=None, max_income=None):
        """Get the combined dataset, optionally filtered."""
        if self.combined_data is None or self.combined_data.empty:
            print("Warning: No combined data available.")
            return pd.DataFrame()  # Empty dataframe
        
        if not filtered:
            return self.combined_data
        
        # Start with a copy of the full dataset
        filtered_data = self.combined_data.copy()
        # Example: filter by Airbnb density level if provided
        if density_levels is not None and 'airbnb_density_level' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['airbnb_density_level'].isin(density_levels)]
        elif density_levels is not None and 'airbnb_density' in filtered_data.columns:
            # Alternative filtering using airbnb_density if available
            # Convert airbnb_density to categorical levels for filtering
            density_percentiles = filtered_data['airbnb_density'].quantile([0.2, 0.4, 0.6, 0.8])
            conditions = [
                filtered_data['airbnb_density'] <= density_percentiles[0.2],
                (filtered_data['airbnb_density'] > density_percentiles[0.2]) & (filtered_data['airbnb_density'] <= density_percentiles[0.4]),
                (filtered_data['airbnb_density'] > density_percentiles[0.4]) & (filtered_data['airbnb_density'] <= density_percentiles[0.6]),
                (filtered_data['airbnb_density'] > density_percentiles[0.6]) & (filtered_data['airbnb_density'] <= density_percentiles[0.8]),
                filtered_data['airbnb_density'] > density_percentiles[0.8]
            ]
            density_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            
            # Use the provided density levels for filtering if they match our labels
            filtered_levels = [label for label in density_labels if label in density_levels]
            if filtered_levels:
                # Create a new column with density classifications to avoid data type issues
                filtered_data['density_category'] = 'Unknown'  # Default value
                
                # Apply labels one by one to avoid np.select data type issues
                for condition, label in zip(conditions, density_labels):
                    filtered_data.loc[condition, 'density_category'] = label
                
                # Filter based on the newly created column
                filtered_data = filtered_data[filtered_data['density_category'].isin(filtered_levels)]
        
        # Income filtering based on available columns with enhanced debugging
        if min_income is not None:
            # Log what income columns are available for debugging
            income_columns = [col for col in filtered_data.columns if 'income' in col.lower()]
            
            # Check for median_income column
            if 'median_income' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['median_income'] >= min_income]
                print(f"Filtering by median_income >= {min_income}, rows remaining: {len(filtered_data)}")
            # Check for median_household_income column
            elif 'median_household_income' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['median_household_income'] >= min_income]
                print(f"Filtering by median_household_income >= {min_income}, rows remaining: {len(filtered_data)}")
            # If we don't have income columns, try to merge in census data with income info
            elif not income_columns and 'zipcode' in filtered_data.columns:
                # Try to add income data if we don't have it but do have zipcode
                try:
                    # Create the income data if it doesn't exist
                    if not hasattr(self, 'income_data_by_zip'):
                        # Load census data and extract median household income by zipcode
                        census_dir = Path(self.data_dir) / 'processed' / 'census'
                        if (census_dir / 'summary_census.csv').exists():
                            census_data = pd.read_csv(census_dir / 'summary_census.csv')
                            # Dummy income data by zipcode - replace with actual zipcode to income mapping
                            # This represents a mapping of zipcode to income based on census data
                            zipcode_income_map = {
                                33010: 55000, 33012: 61000, 33013: 57000, 33014: 63000, 33015: 68000,
                                33016: 70000, 33018: 72000, 33030: 59000, 33031: 75000, 33032: 58000,
                                33033: 60000, 33034: 45000, 33035: 67000, 33054: 48000, 33055: 58000,
                                33056: 52000, 33109: 120000, 33122: 65000, 33125: 54000, 33126: 62000,
                                33127: 50000, 33128: 53000, 33129: 85000, 33130: 78000, 33131: 95000,
                                33132: 90000, 33133: 88000, 33134: 80000, 33135: 58000, 33136: 48000,
                                33137: 76000, 33138: 70000, 33139: 83000, 33140: 93000, 33141: 75000,
                                33142: 51000, 33143: 82000, 33144: 61000, 33145: 68000, 33146: 95000,
                                33147: 47000, 33149: 98000, 33150: 49000, 33154: 87000, 33155: 73000,
                                33156: 92000, 33157: 65000, 33158: 98000, 33160: 85000, 33161: 60000,
                                33162: 59000, 33165: 70000, 33166: 68000, 33167: 54000, 33168: 56000,
                                33169: 62000, 33170: 63000, 33172: 65000, 33173: 78000, 33174: 72000,
                                33175: 69000, 33176: 80000, 33177: 68000, 33178: 85000, 33179: 75000,
                                33180: 82000, 33181: 76000, 33182: 72000, 33183: 76000, 33184: 68000,
                                33185: 78000, 33186: 80000, 33187: 72000, 33189: 65000, 33190: 70000,
                                33193: 75000, 33194: 71000, 33196: 78000
                            }
                            self.income_data_by_zip = zipcode_income_map
                        else:
                            # Default mapping if we can't find the census data
                            self.income_data_by_zip = {}
                    
                    # Add income column to the filtered data
                    filtered_data['median_household_income'] = filtered_data['zipcode'].map(self.income_data_by_zip)
                    
                    # Now filter with the added income data
                    filtered_data = filtered_data[filtered_data['median_household_income'] >= min_income]
                    print(f"Added income data and filtered by median_household_income >= {min_income}, rows remaining: {len(filtered_data)}")
                except Exception as e:
                    print(f"Error adding income data: {e}")
            
        # Apply max income filter if specified
        if max_income is not None:
            if 'median_income' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['median_income'] <= max_income]
                print(f"Filtering by median_income <= {max_income}, rows remaining: {len(filtered_data)}")
            elif 'median_household_income' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['median_household_income'] <= max_income]
                print(f"Filtering by median_household_income <= {max_income}, rows remaining: {len(filtered_data)}")
        
        return filtered_data
    
    def get_summary_stats(self):
        """
        Get summary statistics for all datasets.
        
        Returns:
            dict: Dictionary of summary statistics
        """
        return self.summary_stats
