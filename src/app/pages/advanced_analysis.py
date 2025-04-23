#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Data Mining Analysis Page

This module contains the Streamlit interface for advanced data mining
visualizations and insights for the Miami Housing Impact Hub.
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import importlib.util

# Import core dependencies and utilities
from ..utils.data_loader import DataLoader
from ..utils.helpers import ensure_data_loaded, load_config
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import traceback
import logging

# Import models with try/except to handle missing dependencies
try:
    from src.models.advanced_mining import NeighborhoodSegmentation, RandomForestPredictor, MonteCarloInvestmentSimulator
    advanced_mining_available = True
except ImportError as e:
    logging.error(f"Error importing advanced_mining: {e}", exc_info=True)
    advanced_mining_available = False

try:
    from src.models.time_series_analysis import TimeSeriesDecomposer
    time_series_available = True
except ImportError as e:
    logging.error(f"Error importing time_series_analysis: {e}", exc_info=True)
    time_series_available = False

try:
    from src.models.association_rules import AmenityAssociationMiner
    association_rules_available = True
except ImportError as e:
    logging.error(f"Error importing association_rules: {e}", exc_info=True)
    association_rules_available = False

# Check if geospatial packages are available
geospatial_available = False
try:
    import pysal
    import contextily as ctx
    from src.models.geospatial_analysis import GeospatialHotspotAnalysis
    geospatial_available = True
except ImportError as e:
    logging.error(f"Error importing geospatial dependencies: {e}", exc_info=True)

# Import visualization functions with try/except blocks
try:
    from src.visualization.association_plots import plot_association_rules
    plots_association_available = True
except ImportError as e:
    logging.error(f"Error importing association plot functions: {e}", exc_info=True)
    plots_association_available = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_advanced_analysis_page(data_loader, agent):
    """
    Display the advanced data mining analysis page.
    
    This page provides access to sophisticated data mining insights
    like k-means clustering, predictive modeling, and more.
    
    Args:
        data_loader: DataLoader instance with access to all datasets
        agent: HousingImpactAgent instance for proactive insights
    """
    st.header("Advanced Data Mining Analysis")
    
    # Explanation of this page
    st.markdown("""
        This section provides advanced data mining insights derived from Miami-Dade housing and Airbnb data.
        These analyses go beyond basic statistics to identify patterns, predict trends, and suggest 
        strategies for different stakeholders.
    """)
    
    # Create tabs for different analyses based on available features
    tab_titles = [
        "Neighborhood Segmentation", 
        "Price Prediction", 
        "Investment Simulation",
    ]
    
    # Only add tabs for available features
    if time_series_available:
        tab_titles.append("Time Series Decomposition")
    
    if association_rules_available:
        tab_titles.append("Association Rule Mining")
    
    # Add Geospatial Hotspot Analysis as "Coming Soon" if dependencies are missing
    if geospatial_available:
        tab_titles.append("Geospatial Hotspots")
    else:
        tab_titles.append("Geospatial Hotspots (Coming Soon)")
    
    tab_titles.append("Custom Analysis")
    
    tabs = st.tabs(tab_titles)
    
    # Map tabs to functions
    function_map = {
        tab_titles[0]: show_neighborhood_segmentation,
        tab_titles[1]: show_price_prediction,
        tab_titles[2]: show_investment_simulation,
    }
    
    # Add optional features to function map
    current_index = 3
    if time_series_available:
        function_map[tab_titles[current_index]] = show_time_series_decomposition
        current_index += 1
    
    if association_rules_available:
        function_map[tab_titles[current_index]] = show_association_rules_analysis
        current_index += 1
    
    if geospatial_available:
        function_map[tab_titles[current_index]] = show_geospatial_hotspot_analysis
        current_index += 1
    else:
        function_map[tab_titles[current_index]] = show_feature_coming_soon
        current_index += 1
    
    function_map[tab_titles[-1]] = show_custom_analysis_interface

    # Execute the function associated with each tab
    for tab, title in zip(tabs, tab_titles):
        with tab:
            if title == "Custom Analysis": # Custom analysis doesn't need data_loader
                 function_map[title]()
            elif title in function_map:
                try:
                    function_map[title](data_loader, agent)
                except Exception as e:
                    st.error(f"An error occurred in the '{title}' section.")
                    logger.error(f"Error in {title}: {e}\n{traceback.format_exc()}")
            else:
                 st.warning(f"Analysis '{title}' is not yet implemented.")


def show_time_series_decomposition(data_loader):
    """Displays time series decomposition analysis for ZORI.

    Args:
        data_loader: DataLoader instance.
    """
    st.subheader("Time Series Decomposition of Rental Prices (ZORI)")
    st.markdown("""
    This analysis decomposes the Zillow Observed Rent Index (ZORI) for the Miami-Dade area 
    into its underlying components: trend, seasonality, and residual (noise). 
    Understanding these components helps identify long-term price movements, 
    cyclical patterns, and irregular fluctuations.
    """)

    try:
        # Load ZORI data
        zori_df = data_loader.get_zori_data()
        if zori_df is None or zori_df.empty:
            st.warning("ZORI data could not be loaded. Cannot perform time series analysis.")
            return
        
        # --- Data Preprocessing --- 
        # Ensure 'Date' column exists and is datetime
        if 'Date' not in zori_df.columns:
             st.error("ZORI DataFrame must contain a 'Date' column.")
             return
        zori_df['Date'] = pd.to_datetime(zori_df['Date'], errors='coerce')
        zori_df = zori_df.dropna(subset=['Date'])
        
        # Ensure 'ZORI' column exists and is numeric
        if 'ZORI' not in zori_df.columns:
            st.error("ZORI DataFrame must contain a 'ZORI' column.")
            return
        zori_df['ZORI'] = pd.to_numeric(zori_df['ZORI'], errors='coerce')
        
        # Set index, sort, and select the series
        zori_df = zori_df.set_index('Date').sort_index()
        time_series = zori_df['ZORI'].dropna() # Drop NaNs before decomposition
        
        if time_series.empty:
            st.warning("No valid ZORI data points found after preprocessing.")
            return
            
        # Display raw data preview
        st.markdown("#### ZORI Time Series Preview")
        st.line_chart(time_series)
        
        with st.expander("View Raw ZORI Data"):
            st.dataframe(time_series)

        # --- Analysis Configuration --- 
        st.markdown("#### Decomposition Configuration")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Decomposition Model", ('additive', 'multiplicative'), index=0,
                                      help="'additive' assumes seasonal effects are constant, 'multiplicative' assumes they scale with the trend.")
        with col2:
            # Infer frequency or default to monthly (12)
            freq = pd.infer_freq(time_series.index)
            default_period = 12 if freq and ('M' in freq or 'm' in freq) else 4 # Default quarterly if not monthly
            if len(time_series) < default_period * 2:
                 default_period = max(2, len(time_series) // 2) # Adjust if series too short
            
            period = st.number_input("Seasonality Period", min_value=2, 
                                     max_value=max(24, len(time_series)//2), 
                                     value=default_period, step=1, 
                                     help="The number of observations per seasonal cycle (e.g., 12 for monthly data with annual seasonality).")

        # --- Run Analysis --- 
        if st.button("Decompose Time Series"):
            if len(time_series) < period * 2:
                 st.warning(f"Time series is too short for the selected period ({period}). Needs at least {period*2} data points.")
            else:
                with st.spinner("Performing decomposition..."):
                    try:
                        decomposer = TimeSeriesDecomposer(time_series)
                        success = decomposer.decompose(model=model_type, period=period)

                        if success:
                            st.success("Decomposition successful!")
                            fig = decomposer.plot_decomposition(title=f"ZORI Time Series Decomposition ({model_type.capitalize()})")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Interpretation guidance
                            st.markdown("**Interpretation:**")
                            st.markdown("""
                            - **Observed:** The original ZORI data.
                            - **Trend:** The long-term progression of rental prices, ignoring seasonality and noise.
                            - **Seasonal:** Patterns that repeat over the specified period (e.g., yearly cycles).
                            - **Residual:** The leftover noise or irregular component after removing trend and seasonality.
                            """)
                        else:
                            st.warning("Decomposition could not be performed. The time series might be too short, have too many missing values, or other issues. Check logs.")
                    
                    except Exception as e:
                        st.error("An error occurred during decomposition.")
                        logger.error(f"Time Series Decomposition failed: {e}\n{traceback.format_exc()}")

    except FileNotFoundError:
        st.error("ZORI data file not found. Please ensure 'zori_data.csv' is in the processed data directory.")
    except Exception as e:
        st.error("An unexpected error occurred while loading or processing ZORI data.")
        logger.error(f"Error in time series decomposition section: {e}\n{traceback.format_exc()}")


def show_association_rules_analysis(data_loader):
    st.subheader("Association Rule Mining for Listing Amenities")
    st.markdown("Discover relationships between listing amenities and characteristics like price range or rating.")
 
    listings_df = data_loader.get_data('listings_processed')
    if listings_df is None or listings_df.empty:
        st.error("Processed listings data is not available. Please run the data processing pipeline.")
        return
 
    # --- Configuration --- 
    st.sidebar.subheader("Association Rules Settings")
    target_options = ['price', 'review_scores_rating'] # Add other potential numeric columns if needed
    target_col = st.sidebar.selectbox("Target Variable (to discretize)", target_options, index=0,
                                    help="Select the variable to find associations with (e.g., high price, good rating).")
 
    min_support = st.sidebar.slider("Minimum Support", 0.001, 0.1, 0.01, 0.001, format="%.3f",
                                   help="Minimum frequency of an itemset in the dataset.")
    metric = st.sidebar.selectbox("Rule Metric", ['lift', 'confidence'], index=0,
                                help="Metric used to evaluate the strength of the association.")
    min_threshold = st.sidebar.slider(f"Minimum {metric.capitalize()}", 0.1, 2.0, 1.0 if metric == 'lift' else 0.5, 0.1,
                                     help=f"Minimum threshold for the selected rule metric ({metric}).")
    n_bins = st.sidebar.slider("Number of Target Bins", 2, 5, 3, 1,
                              help="How many categories to divide the target variable into (e.g., Low/Med/High Price).")
    min_amenity_freq = st.sidebar.slider("Minimum Amenity Frequency", 5, 100, 10, 5,
                                        help="Minimum number of listings an amenity must appear in to be included.")
 
    # --- Analysis Trigger --- 
    if st.button("Run Association Rule Mining"):
        st.info("Running analysis... This might take a moment.")
        progress_bar = st.progress(0)
        status_text = st.empty()
 
        try:
            status_text.text("Initializing Miner...")
            miner = AmenityAssociationMiner(listings_df)
            progress_bar.progress(10)
 
            status_text.text("Preprocessing data (parsing amenities, discretizing target, one-hot encoding)...")
            start_time = time.time()
            transaction_df = miner.preprocess_data(
                target_col=target_col,
                n_bins=n_bins,
                min_amenity_freq=min_amenity_freq
            )
            preprocess_time = time.time() - start_time
            status_text.text(f"Preprocessing complete ({preprocess_time:.2f}s). Found {transaction_df.shape[1]} items.")
            progress_bar.progress(50)
 
            if transaction_df is None or transaction_df.empty:
                st.warning("Preprocessing failed or resulted in no data. Check logs or adjust parameters (e.g., min amenity frequency).")
                st.stop()
 
            status_text.text(f"Finding frequent itemsets (min_support={min_support})...")
            start_time = time.time()
            # Adjust metric/threshold based on selection for find_rules
            min_conf = min_threshold if metric == 'confidence' else 0.1 # Use default confidence if metric is lift
            min_lift = min_threshold if metric == 'lift' else 1.0 # Use default lift if metric is confidence
            rules = miner.find_rules(
                transaction_df,
                min_support=min_support,
                min_confidence=min_conf, # Pass appropriate confidence threshold
                metric=metric,
                min_lift=min_lift # Pass appropriate lift threshold
            )
            mining_time = time.time() - start_time
            status_text.text(f"Rule mining complete ({mining_time:.2f}s).")
            progress_bar.progress(90)
 
            if rules is None:
                st.error("An error occurred during rule mining. Check logs.")
                st.stop()
            elif rules.empty:
                st.warning(f"No association rules found meeting the specified criteria (min_support={min_support}, min_{metric}={min_threshold}). Try relaxing the thresholds.")
            else:
                st.success(f"Found {len(rules)} association rules.")
 
                # --- Display Results ---
                st.subheader("Discovered Association Rules")
 
                # Filter for rules where consequent is a target category bin
                target_bin_prefix = f"{target_col}_Bin"
                rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                target_rules = rules[rules['consequents_str'].str.startswith(target_bin_prefix)]
 
                if target_rules.empty:
                    st.info("No rules found with the target variable as the consequent. Displaying all rules.")
                    display_rules = rules
                else:
                    st.info(f"Showing {len(target_rules)} rules with '{target_col}' bins as the consequent.")
                    display_rules = target_rules
 
                # Format frozensets for display
                display_rules['antecedents_str'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                # display_rules['consequents_str'] is already created
 
                st.dataframe(display_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].round(4))
 
                st.subheader("Rules Visualization")
                status_text.text("Generating plot...")
                rules_plot = plot_association_rules(display_rules, metric=metric)
                if rules_plot:
                    st.pyplot(rules_plot)
                else:
                    st.warning("Could not generate the rules visualization plot.")
 
            progress_bar.progress(100)
            status_text.text("Analysis complete.")
 
        except Exception as e:
            logger.error(f"Error during Association Rule Mining analysis: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {e}")
            status_text.text("Analysis failed.")
            if progress_bar: progress_bar.progress(100)
        finally:
            # Ensure spinner/progress stops
            pass
 
 
def show_custom_analysis_interface():
    """Interface for users to request custom analyses."""
    st.subheader("Custom Analysis Request")
    
    st.markdown("""
        Need a specific data mining analysis for your decision-making? 
        Submit your request here, and our data science team will evaluate it.
    """)
    
    analysis_type = st.selectbox(
        "Analysis Type",
        options=[
            "Select an analysis type...",
            "Clustering (K-means, DBSCAN, etc.)",
            "Regression/Prediction",
            "Classification",
            "Time Series Analysis",
            "Association Rules Mining",
            "Text Mining/NLP",
            "Other (please specify)"
        ]
    )
    
    business_goal = st.text_area(
        "Business Goal",
        placeholder="What decision or insight are you trying to gain from this analysis?"
    )
    
    data_needed = st.multiselect(
        "Data Sources Needed",
        options=[
            "Airbnb listings", 
            "Property sales", 
            "Census demographics",
            "Rental rates",
            "Crime statistics",
            "Economic indicators",
            "Geographic/spatial data",
            "Other (please specify)"
        ]
    )
    
    additional_notes = st.text_area(
        "Additional Notes",
        placeholder="Any other details or requirements for your analysis?"
    )
    
    if st.button("Submit Analysis Request"):
        if analysis_type == "Select an analysis type..." or not business_goal:
            st.error("Please select an analysis type and describe your business goal.")
        else:
            st.success("Your analysis request has been submitted! We'll evaluate it for inclusion in our roadmap.")
            
            # In a real application, this would be logged or emailed to the development team
            # Here, we're just displaying a confirmation message
            with st.expander("Request Details", expanded=True):
                st.write(f"**Analysis Type:** {analysis_type}")
                st.write(f"**Business Goal:** {business_goal}")
                st.write(f"**Data Sources Needed:** {', '.join(data_needed)}")
                st.write(f"**Additional Notes:** {additional_notes}")

@ensure_data_loaded(datasets=['neighborhood_stats', 'listings_processed'])
def show_neighborhood_segmentation(data_loader):
    st.subheader("Neighborhood Segmentation using K-Means Clustering")
    st.markdown("Group similar neighborhoods based on statistical features.")

    # --- Configuration ---
    st.sidebar.subheader("Segmentation Settings")
    max_clusters = st.sidebar.slider("Maximum Number of Clusters (K)", 2, 10, 5, 1, help="The maximum number of clusters to consider for the optimal K.")
    cluster_features = st.sidebar.multiselect("Features for Clustering", ['price', 'bedrooms', 'bathrooms', 'sqft', 'latitude', 'longitude'], default=['price', 'bedrooms', 'bathrooms', 'sqft'])
    st.sidebar.write("Note: Latitude and Longitude are used for spatial clustering.")

    # --- Data Preparation ---
    try:
        neighborhood_df = data_loader.get_data('neighborhood_stats')
        if neighborhood_df is None or neighborhood_df.empty:
            st.error("Neighborhood statistics data is not available. Please run the data processing pipeline.")
            return

        # Select features for clustering
        features_df = neighborhood_df[cluster_features]
        if features_df is None or features_df.empty:
            st.error("No features selected for clustering or features not found in data.")
            return

        # Scale features using StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        scaled_features_df = pd.DataFrame(scaled_features, columns=features_df.columns, index=features_df.index)

        # Initialize the segmentation model
        segmentation = NeighborhoodSegmentation(scaled_features_df)

        # Find the optimal number of clusters (K)
        optimal_k = segmentation.find_optimal_k(max_k=max_clusters)

        # Perform K-Means clustering with the optimal K
        cluster_labels = segmentation.perform_clustering(n_clusters=optimal_k)

        # Add cluster labels to the original DataFrame for visualization/interpretation
        neighborhood_df['cluster'] = cluster_labels

        # --- Display Results ---
        st.subheader("Clustering Results")
        st.dataframe(neighborhood_df[['neighborhood', 'cluster']].sort_values('cluster'))

        st.subheader("Cluster Visualization")
        cluster_plot = plot_clusters(neighborhood_df, cluster_features[0], cluster_features[1], 'cluster')
        if cluster_plot:
            st.pyplot(cluster_plot)
        else:
            st.warning("Could not generate the cluster plot.")

    except Exception as e:
        logger.error(f"Error during Neighborhood Segmentation analysis: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")


@ensure_data_loaded(datasets=['listings_processed'])
def show_price_prediction(data_loader):
    st.subheader("Airbnb Price Prediction using Random Forest")
    st.markdown("Predict listing prices based on various features.")

    listings_df = data_loader.get_data('listings_processed')
    if listings_df is None or listings_df.empty:
        st.error("Processed listings data is not available. Please run the data processing pipeline.")
        return

    # --- Feature Selection and Preparation ---
    selected_features = st.sidebar.multiselect("Features for Prediction", ['host_is_superhost', 'host_listings_count', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities'], default=['host_is_superhost', 'host_listings_count', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds'])
    test_split = st.sidebar.slider("Test Data Split", 0.1, 0.5, 0.2, 0.1, help="The proportion of data to use for testing.")

    # Initialize the predictor model
    predictor = RandomForestPredictor(listings_df)

    # Prepare the data for training and testing
    X_train, X_test, y_train, y_test = predictor.prepare_data(target_column='price', feature_columns=selected_features, test_size=test_split, random_state=42)

    # Train the model
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 100, 50, 10, help="The number of trees in the forest.")
    max_depth = st.sidebar.slider("Maximum Depth", 5, 20, 10, 1, help="The maximum depth of each tree.")
    predictor.train_model(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Make predictions on the test set
    y_pred = predictor.predict(X_test)

    # Evaluate the model
    mse, r2 = predictor.evaluate(y_test, y_pred)

    # --- Display Results ---
    st.subheader("Model Evaluation")
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
    st.metric(label="R-squared (R2)", value=f"{r2:.4f}")

    st.subheader("Feature Importance")
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        importance_plot = plot_feature_importance(importance_df)
        if importance_plot:
            st.pyplot(importance_plot)
        else:
            st.warning("Could not generate feature importance plot.")
    else:
        st.warning("Could not retrieve feature importances.")


@ensure_data_loaded(datasets=['listings_processed', 'zillow_processed'])
def show_investment_simulation(data_loader):
    st.subheader("Investment Property ROI Simulation using Monte Carlo")
    st.markdown("Simulate potential return on investment for properties based on historical trends and market volatility.")

    zori_df = data_loader.get_data('zillow_processed')
    if zori_df is None or zori_df.empty:
        st.error("Zillow data is not available. Please run the data processing pipeline.")
        return

    # --- Data Preparation & Simulation Setup ---
    initial_investment = st.sidebar.number_input("Initial Investment", min_value=10000, value=50000, step=1000, help="The initial amount to invest.")
    n_years = st.sidebar.slider("Holding Period (Years)", 1, 10, 5, 1, help="The number of years to hold the investment.")
    n_simulations = st.sidebar.slider("Number of Simulations", 100, 1000, 500, 100, help="The number of simulations to run.")
    purchase_price = st.sidebar.number_input("Purchase Price", min_value=10000, value=200000, step=1000, help="The price at which to purchase the property.")
    annual_expenses = st.sidebar.number_input("Annual Expenses", min_value=0, value=5000, step=1000, help="The annual expenses for the property (e.g., maintenance, taxes).")

    # Initialize the simulator
    simulator = MonteCarloInvestmentSimulator(zori_df, 'ZORI_NSA')

    # Run the simulation
    results = simulator.simulate_investment(initial_investment=initial_investment, holding_period_years=n_years, num_simulations=n_simulations, purchase_price=purchase_price, annual_expenses=annual_expenses)

    # --- Display Results ---
    if results:
        st.subheader("Simulation Results")
        summary_stats = results['summary_stats']

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Final Value", f"${summary_stats['mean_final_value']:.2f}")
        col2.metric("Median ROI", f"{summary_stats['median_roi']:.2%}")
        col3.metric("Standard Deviation of ROI", f"{summary_stats['std_roi']:.2%}")

        st.subheader("Simulated Investment Paths")
        paths_df = results['simulation_paths']
        if paths_df is not None:
            paths_plot = plot_monte_carlo_paths(paths_df)
            if paths_plot:
                st.plotly_chart(paths_plot, use_container_width=True)
            else:
                st.warning("Could not generate simulation paths plot.")
        else:
            st.warning("Simulation paths data is not available.")

        st.subheader("Distribution of Final Investment Values")
        final_values = paths_df.iloc[-1] if paths_df is not None else None
        if final_values is not None:
            hist_plot = plot_monte_carlo_histogram(final_values)
            if hist_plot:
                st.plotly_chart(hist_plot, use_container_width=True)
            else:
                st.warning("Could not generate final values histogram.")
        else:
            st.warning("Final values data is not available for histogram.")
    else:
        st.warning("Simulation did not produce valid results.")


@ensure_data_loaded(datasets=['zillow_processed'])
def show_time_series_decomposition(data_loader):
    st.subheader("Time Series Decomposition for Zillow Observed Rent Index (ZORI)")
    st.markdown("Analyze the trend, seasonal, and residual components of ZORI data over time.")

    zori_df = data_loader.get_data('zillow_processed')
    if zori_df is None or zori_df.empty:
        st.error("Zillow data is not available. Please run the data processing pipeline.")
        return

    # --- Data Preparation ---
    selected_region = st.selectbox("Select Region", zori_df['RegionName'].unique())
    region_data = zori_df[zori_df['RegionName'] == selected_region].sort_index()
    if region_data.empty:
        st.warning(f"No ZORI data found for the selected region: {selected_region}")
        return

    # Prepare the series for decomposition
    target_column = 'ZORI_NSA'
    if target_column not in region_data.columns:
        st.error(f"Target column '{target_column}' not found in the data for {selected_region}.")
        return

    time_series = region_data[target_column].dropna() # Drop NA values
    if time_series.empty:
        st.warning(f"No valid time series data for '{target_column}' in {selected_region} after dropping NA.")
        return

    # Initialize the decomposer
    decomposer = TimeSeriesDecomposer(time_series)

    # --- Analysis Configuration ---
    st.markdown("#### Decomposition Configuration")
    col1, col2 = st.columns(2)
    with col1:
        decomposition_model = st.selectbox("Decomposition Model", ('additive', 'multiplicative'), index=0, help="'additive' assumes seasonal effects are constant, 'multiplicative' assumes they scale with the trend.")
    with col2:
        seasonal_period = st.number_input("Seasonality Period", min_value=2, value=12, step=1, help="The number of observations per seasonal cycle (e.g., 12 for monthly data with annual seasonality).")

    # --- Run Analysis ---
    if st.button("Decompose Time Series"):
        if len(time_series) < seasonal_period * 2:
            st.warning(f"Time series is too short for the selected period ({seasonal_period}). Needs at least {seasonal_period*2} data points.")
        else:
            with st.spinner("Performing decomposition..."):
                try:
                    decomposition_result = decomposer.decompose(model=decomposition_model, period=seasonal_period)

                    if decomposition_result:
                        st.success("Decomposition successful!")
                        decomp_plot = plot_time_series_decomposition(decomposition_result)
                        if decomp_plot:
                            st.plotly_chart(decomp_plot, use_container_width=True)
                        else:
                            st.warning("Could not generate decomposition plot.")
                    else:
                        st.warning("Decomposition could not be performed. The time series might be too short, have too many missing values, or other issues. Check logs.")
                except Exception as e:
                    st.error("An error occurred during decomposition.")
                    logger.error(f"Time Series Decomposition failed: {e}\n{traceback.format_exc()}")
