#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive Dashboard Page Module

This module contains the code for the application's interactive dashboard page,
which provides visualizations of Airbnb density and housing affordability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def show_dashboard_page(data_loader, agent):
    """
    Display the interactive dashboard page.
    
    This page provides interactive visualizations of Airbnb density
    and housing affordability metrics across Miami-Dade neighborhoods.
    
    Args:
        data_loader: DataLoader instance with access to all datasets
        agent: HousingImpactAgent instance for proactive insights
    """
    # Use columns to make better use of wide layouts
    # Note: set_page_config must be called in app.py as it must be the first Streamlit command
    st.header("Interactive Dashboard")
    
    st.markdown("""
        Explore the relationship between short-term rentals and housing affordability
        across Miami-Dade County neighborhoods. Use the filters in the sidebar to
        customize your view.
    """)
    
    # Get and display an agent insight related to the dashboard data
    with st.sidebar.container(border=True):
        st.sidebar.markdown("### Agent Insight")
        # Get a relevant insight from the agent
        insights = agent.get_proactive_insights()
        if insights:
            selected_insight = insights[0]  # Just pick the first one for simplicity
            st.sidebar.markdown(f"**{selected_insight['title']}**")
            st.sidebar.markdown(selected_insight['description'])
            
            # Add a button to ask a follow-up question
            if st.sidebar.button("Ask about this insight"):
                st.session_state['page'] = "Housing Assistant"
                st.session_state['user_question'] = f"Tell me more about {selected_insight['title'].lower()}"
                st.rerun()
    
    # Sidebar filters
    st.sidebar.subheader("Dashboard Filters")
    
    # Only show filters if we have data
    if data_loader.combined_data is not None:
        # Filter for Airbnb density levels
        density_options = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        selected_density = st.sidebar.multiselect(
            "Airbnb Density Levels:",
            options=density_options,
            default=density_options
        )
        
        # Income range filter - handle potentially missing columns
        if 'median_household_income' in data_loader.combined_data.columns:
            min_income = int(data_loader.combined_data['median_household_income'].min())
            max_income = int(data_loader.combined_data['median_household_income'].max())
            income_range = st.sidebar.slider(
                "Median Household Income Range ($):",
                min_value=min_income,
                max_value=max_income,
                value=(min_income, max_income)
            )
        else:
            # Default values for prototyping if column doesn't exist
            income_range = st.sidebar.slider(
                "Median Household Income Range ($):",
                min_value=30000,
                max_value=120000,
                value=(30000, 120000)
            )
        
        # Prepare filter parameters
        filter_params = {
            'density_levels': selected_density if selected_density else density_options
        }
        
        # Only add income filters if the column exists
        if 'median_household_income' in data_loader.combined_data.columns:
            filter_params['min_income'] = income_range[0]
            filter_params['max_income'] = income_range[1]
        
        # Start with loading the data
        combined_data = data_loader.get_combined_data()
        
        # Check if we have data to display
        if not combined_data.empty:
            # Apply filters to get a filtered dataset
            filtered_data = data_loader.get_combined_data(
                filtered=True,
                **filter_params
            )
            
            # Add a notification to show that filters are applied
            if selected_density and len(selected_density) < len(density_options):
                st.sidebar.success(f"✓ Filters applied! Showing {len(filtered_data)} of {len(combined_data)} neighborhoods.")
            
            # If the filtering reduced the data too much, warn the user
            if len(filtered_data) < 3 and len(combined_data) > 3:
                st.sidebar.warning("⚠️ Current filters are very restrictive. Consider relaxing them to see more data.")
                
            # Add a reset button for filters
            if st.sidebar.button("Reset All Filters"):
                # Clear filters by updating session state
                st.session_state['dashboard_filters_reset'] = True
                st.rerun()
                
            # Clear the reset flag if it exists
            if 'dashboard_filters_reset' in st.session_state and st.session_state['dashboard_filters_reset']:
                st.session_state['dashboard_filters_reset'] = False
            
            # Important: Use the filtered data instead of the combined data for visualizations
            # Replace combined_data with filtered_data in all subsequent function calls
            combined_data = filtered_data
            
            # Prepare interaction data for agent
            interaction_data = {
                'page': 'Interactive Dashboard',
                'filters': {
                    'density_levels': selected_density if selected_density else density_options
                }
            }
            
            # Only add income filters if the column exists
            if 'median_household_income' in data_loader.combined_data.columns:
                interaction_data['filters']['min_income'] = income_range[0]
                interaction_data['filters']['max_income'] = income_range[1]
            
            # Update agent with filter selections
            agent.update_from_user_interaction(interaction_data)
    else:
        # Show warning if no data available
        st.warning("No combined data available. Please ensure miami_dade_merged_data.csv is present in the processed data directory.")
        return
    
    # Create a tabbed interface for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Airbnb Distribution", "Affordability Metrics", "Rental Analysis", "Geographic Insights"])
    
    with tab1:
        # Pass the filtered_data to ensure filters are applied
        show_airbnb_distribution(data_loader, filtered_data)
    
    with tab2:
        # Pass the filtered_data to ensure filters are applied
        show_affordability_metrics(data_loader, filtered_data)
    
    with tab3:
        # The rental analysis doesn't use the combined data directly
        show_rental_analysis(data_loader)
    
    with tab4:
        # Pass the filtered_data to ensure filters are applied
        show_geographic_insights(data_loader, filtered_data)

def show_airbnb_distribution(data_loader, combined_data):
    """
    Show visualizations related to Airbnb distribution.
    
    Args:
        data_loader: DataLoader instance
        combined_data: Combined dataset
    """
    st.subheader("Airbnb Listings Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Listings by neighborhood (using airbnb_count column from miami_dade_merged_data.csv)
        if 'airbnb_count' in combined_data.columns and 'neighborhood' in combined_data.columns:
            # Sort by airbnb count
            sorted_data = combined_data.sort_values('airbnb_count', ascending=False)
            
            # Create the bar chart
            fig = px.bar(
                sorted_data,
                x='neighborhood',
                y='airbnb_count',
                title='Neighborhoods by Airbnb Count',
                color='airbnb_count',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                xaxis_title='Neighborhood',
                yaxis_title='Number of Listings',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Airbnb count data available.")
            try:
                st.image("Presentation/images/airbnb_listings_by_zipcode.png", 
                        caption="Distribution of Airbnb Listings by ZIP Code")
            except:
                st.info("Airbnb distribution visualization not available.")
    
    with col2:
        # Airbnb density visualization
        if 'airbnb_density' in combined_data.columns and 'neighborhood' in combined_data.columns:
            # Sort by airbnb density
            sorted_data = combined_data.sort_values('airbnb_density', ascending=False)
            
            # Create the bar chart
            fig = px.bar(
                sorted_data,
                x='neighborhood',
                y='airbnb_density',
                title='Neighborhoods by Airbnb Density',
                color='airbnb_density',
                color_continuous_scale='Oranges'
            )
            
            fig.update_layout(
                xaxis_title='Neighborhood',
                yaxis_title='Airbnb Density',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Airbnb density data available.")
            try:
                st.image("Presentation/images/airbnb_home_type_by_zipcode.png", 
                        caption="Percentage of Entire Home Listings by ZIP Code")
            except:
                st.info("Property type distribution visualization not available.")
    
    # Simplified density map
    st.subheader("Airbnb Density Map")
    
    try:
        st.image("Presentation/images/airbnb_density_map.png", 
                caption="Airbnb Rental Density Map of Miami-Dade County")
    except:
        st.info("Density map visualization not available. Select the 'Interactive Dashboard' tab to generate interactive visualizations.")

def show_affordability_metrics(data_loader, combined_data):
    """
    Show visualizations related to housing affordability metrics using the miami_dade_merged_data.csv columns.
    
    Args:
        data_loader: DataLoader instance
        combined_data: Combined dataset
    """
    st.subheader("Housing Affordability Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Property price vs Airbnb price comparison
        if 'median_property_price' in combined_data.columns and 'median_airbnb_price' in combined_data.columns:
            # Create a comparison chart of property prices vs airbnb prices
            fig = px.scatter(
                combined_data,
                x='median_property_price',
                y='median_airbnb_price',
                color='airbnb_density',
                hover_name='neighborhood',
                title='Property Prices vs Airbnb Prices',
                color_continuous_scale='RdYlGn_r'  # Red-Yellow-Green reversed (high density = red)
            )
            
            fig.update_layout(
                xaxis_title='Median Property Price ($)',
                yaxis_title='Median Airbnb Price ($)',
                height=400
            )
            
            # Add a simple trend line without using statsmodels
            try:
                # Try using statsmodels if available
                import statsmodels.api as sm
                
                # Add a trend line with proper error handling
                fig.add_traces(
                    px.scatter(
                        combined_data, 
                        x='median_property_price', 
                        y='median_airbnb_price',
                        trendline='ols'
                    ).data[1]
                )
            except (ImportError, ModuleNotFoundError, IndexError):
                # Calculate a simple linear trendline manually
                x = combined_data['median_property_price']
                y = combined_data['median_airbnb_price']
                
                # Simple linear regression calculation
                n = len(x)
                if n > 1:  # Need at least 2 points for a line
                    x_mean = x.mean()
                    y_mean = y.mean()
                    
                    # Calculate slope and intercept
                    numerator = ((x - x_mean) * (y - y_mean)).sum()
                    denominator = ((x - x_mean) ** 2).sum()
                    
                    slope = numerator / denominator if denominator != 0 else 0
                    intercept = y_mean - (slope * x_mean)
                    
                    # Create line points
                    x_line = np.array([x.min(), x.max()])
                    y_line = slope * x_line + intercept
                    
                    # Add manual trendline
                    fig.add_trace(
                        go.Scatter(
                            x=x_line, 
                            y=y_line, 
                            mode='lines', 
                            name='Trend',
                            line=dict(color='red', dash='dash')
                        )
                    )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create a price vs density chart if available
            if 'median_airbnb_price' in combined_data.columns and 'airbnb_density' in combined_data.columns:
                fig = px.scatter(
                    combined_data,
                    x='airbnb_density',
                    y='median_airbnb_price',
                    hover_name='neighborhood',
                    title='Airbnb Price vs Density',
                    color='median_airbnb_price',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    xaxis_title='Airbnb Density',
                    yaxis_title='Median Airbnb Price ($)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No price comparison data available.")
    
    with col2:
        # Income vs Rent scatter plot
        if 'median_household_income' in combined_data.columns and 'median_gross_rent' in combined_data.columns:
            # Create the scatter plot
            fig = px.scatter(
                combined_data,
                x='median_household_income',
                y='median_gross_rent',
                color='airbnb_density',
                size='population',
                hover_name='neighborhood',
                color_continuous_scale='RdYlGn_r',
                title='Median Income vs. Median Rent by Neighborhood',
                labels={
                    'median_household_income': 'Median Household Income ($)',
                    'median_gross_rent': 'Median Monthly Rent ($)',
                    'airbnb_density': 'Airbnb Density',
                    'population': 'Population'
                }
            )
            
            fig.update_layout(
                xaxis_title='Median Household Income ($)',
                yaxis_title='Median Monthly Rent ($)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No income vs rent data available.")
    
    # Rent-to-income ratio distribution
    if 'rent_to_income_ratio' in combined_data.columns:
        # Create the histogram
        fig = px.histogram(
            combined_data,
            x='rent_to_income_ratio',
            nbins=20,
            title='Distribution of Rent-to-Income Ratio Across Neighborhoods',
            labels={'rent_to_income_ratio': 'Rent-to-Income Ratio (%)'},
            color_discrete_sequence=['#ff7f0e']
        )
        
        # Add affordability threshold line
        fig.add_vline(x=30, line_dash="dash", line_color="red", 
                     annotation_text="30% Affordability Threshold", 
                     annotation_position="top right")
        
        fig.update_layout(xaxis_title='Rent-to-Income Ratio (%)', 
                         yaxis_title='Number of Neighborhoods')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rent-to-income ratio data available.")

def show_agent_recommendations(agent, data_loader):
    """
    Show agent recommendations based on data analysis.
    
    Args:
        agent: HousingImpactAgent instance
        data_loader: DataLoader instance
    """
    st.subheader("Housing Impact Agent Insights")
    
    st.markdown("""
        Based on the analysis of Miami-Dade housing data and Airbnb listings, the Housing Impact Agent
        provides the following insights and recommendations:
    """)
    
    # Get data structure for agent insights
    data = data_loader.combined_data
    
    if data is not None and not data.empty:
        # Generate insights from the available data
        with st.container(border=True):
            st.markdown("### Airbnb Density Impact")
            
            # Find neighborhood with highest airbnb density
            if 'airbnb_density' in data.columns and 'neighborhood' in data.columns:
                top_density = data.sort_values('airbnb_density', ascending=False).iloc[0]
                st.markdown(f"**{top_density['neighborhood']}** has the highest Airbnb density at **{top_density['airbnb_density']:.3f}**, "
                            f"with {top_density['airbnb_count']} listings for a population of {top_density['population']}.")
            
        with st.container(border=True):
            st.markdown("### Property Price Insights")
            
            # Calculate average pricing information if available
            if 'median_property_price' in data.columns and 'median_airbnb_price' in data.columns:
                avg_property = data['median_property_price'].mean()
                avg_airbnb = data['median_airbnb_price'].mean()
                price_ratio = avg_airbnb * 365 / avg_property * 100  # Annual Airbnb income as % of property price
                
                st.markdown(f"The average property price is **${avg_property:,.0f}** while the average Airbnb price is **${avg_airbnb:.0f}** per night.")
                st.markdown(f"At full occupancy, annual Airbnb income would be approximately **{price_ratio:.1f}%** of the property value.")
        
        # Recommendations section
        st.markdown("### Policy Recommendations")
        
        with st.container(border=True):
            st.markdown("**Targeted Regulation in High-Density Areas**")
            st.markdown("Implement stricter regulations in neighborhoods with the highest Airbnb density to balance tourism benefits and housing affordability.")
        
        with st.container(border=True):
            st.markdown("**Affordability Requirements**")
            st.markdown("Consider requiring Airbnb operators in high-density areas to contribute to affordable housing funds or provide affordable units.")
    else:
        st.info("No data available for agent insights. Please ensure miami_dade_merged_data.csv is available.")

def show_rental_analysis(data_loader):
    """
    Show rental price analysis and actionable insights for stakeholders.
    
    Args:
        data_loader: DataLoader instance with access to rental data
    """
    st.subheader("Rental Market Analysis")
    
    st.markdown("""
        This analysis provides actionable insights into Miami-Dade County's rental market trends,
        helping investors, policymakers, and residents make informed decisions.
    """)
    
    # Try to load rental data by bedroom type
    try:
        rent_by_bedroom_path = "data/processed/rental/avg_rent_by_bedroom.csv"
        rent_by_bedroom_df = pd.read_csv(rent_by_bedroom_path)
        
        # Create two columns for the layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create a dual-axis chart showing rent prices and YoY changes
            fig = go.Figure()
            
            # Convert change percentages to numeric values
            rent_by_bedroom_df['Change Last Year'] = rent_by_bedroom_df['Change Last Year'].replace('No Change', '0%')
            rent_by_bedroom_df['Change Last Year'] = rent_by_bedroom_df['Change Last Year'].str.rstrip('%').astype(float)
            
            # Add bar chart for average rent
            fig.add_trace(go.Bar(
                x=rent_by_bedroom_df['Bedroom Type'],
                y=rent_by_bedroom_df['Average Rent (USD)'],
                name='Average Rent',
                marker_color='#1f77b4',
                text=rent_by_bedroom_df['Average Rent (USD)'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside'
            ))
            
            # Add line chart for YoY change
            fig.add_trace(go.Scatter(
                x=rent_by_bedroom_df['Bedroom Type'],
                y=rent_by_bedroom_df['Change Last Year'],
                name='YoY Change',
                marker_color='#d62728',
                mode='lines+markers+text',
                text=rent_by_bedroom_df['Change Last Year'].apply(lambda x: f"{x}%"),
                textposition='top center',
                yaxis='y2'
            ))
            
            # Setup dual Y-axes
            fig.update_layout(
                title='Average Rent by Bedroom Type with Annual Change',
                xaxis_title='Bedroom Type',
                yaxis_title='Average Rent (USD)',
                yaxis2=dict(
                    title='Year-over-Year Change (%)',
                    overlaying='y',
                    side='right',
                    range=[0, max(rent_by_bedroom_df['Change Last Year']) * 1.2],
                    ticksuffix='%'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Key Insights for Stakeholders")
            
            # Calculate useful metrics
            avg_1bed_rent = rent_by_bedroom_df.loc[rent_by_bedroom_df['Bedroom Type'] == '1 Bedroom', 'Average Rent (USD)'].values[0]
            avg_2bed_rent = rent_by_bedroom_df.loc[rent_by_bedroom_df['Bedroom Type'] == '2 Bedroom', 'Average Rent (USD)'].values[0]
            highest_growth_type = rent_by_bedroom_df.loc[rent_by_bedroom_df['Change Last Year'].idxmax(), 'Bedroom Type']
            highest_growth_rate = rent_by_bedroom_df['Change Last Year'].max()
            
            # Create actionable insights based on data
            with st.container(border=True):
                st.markdown("#### For Investors")
                st.markdown(f"""
                    - **ROI Opportunity**: {highest_growth_type} units show {highest_growth_rate}% annual rent growth, 
                      significantly outperforming other segments
                    - **Premium Calculation**: 2BR units command a {((avg_2bed_rent/avg_1bed_rent)-1)*100:.1f}% premium over 1BR units
                    - **Strategy**: Focus development/acquisition on {highest_growth_type} units 
                      in high-demand neighborhoods to maximize returns
                """)
            
            with st.container(border=True):
                st.markdown("#### For Policymakers")
                st.markdown(f"""
                    - **Affordability Gap**: The average 1BR rent (${avg_1bed_rent:,.0f}) requires an income of 
                      ${avg_1bed_rent*12/0.3:,.0f} to maintain affordability (rent < 30% of income)
                    - **Family Housing Pressure**: The steep price jump to 3BR+ units creates 
                      housing challenges for families
                    - **Policy Approach**: Consider targeted incentives for family-sized rental 
                      development to address supply constraints
                """)
            
            with st.container(border=True):
                st.markdown("#### For Residents")
                st.markdown(f"""
                    - **Budget Planning**: Expect continued rent increases of {rent_by_bedroom_df['Change Last Year'].mean():.1f}% 
                      annually across all unit types
                    - **Value Maximization**: 2BR units offer the best value per bedroom at 
                      ${avg_2bed_rent/2:,.0f}/bedroom vs. ${avg_1bed_rent:,.0f} for 1BR units
                    - **Timing Strategy**: Lock in longer leases for high-growth {highest_growth_type} units to 
                      protect against steep increases
                """)
    
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # For other types of rental analysis if rent_by_bedroom data isn't available
        try:
            rental_data_path = "data/processed/rental/cleaned_rental.csv"
            rental_df = pd.read_csv(rental_data_path)
            
            st.write("Rental data available:", rental_df.columns.tolist())
            
            # Show some basic insights if we have useful columns
            if 'price' in rental_df.columns and 'bedrooms' in rental_df.columns:
                avg_by_bedroom = rental_df.groupby('bedrooms')['price'].mean().reset_index()
                
                fig = px.bar(
                    avg_by_bedroom,
                    x='bedrooms',
                    y='price',
                    title='Average Rental Price by Bedroom Count',
                    labels={'price': 'Average Price ($)', 'bedrooms': 'Number of Bedrooms'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.warning("Rental data files are not available. Please ensure the data is properly loaded.")

def show_geographic_insights(data_loader, combined_data):
    """
    Show geographic insights on housing metrics across Miami-Dade neighborhoods.
    
    Args:
        data_loader: DataLoader instance
        combined_data: Combined dataset
    """
    st.subheader("Geographic Housing Insights")
    
    st.markdown("""
        Explore how housing metrics vary geographically across Miami-Dade County's neighborhoods,
        helping identify opportunity areas and affordability hot spots.
    """)
    
    # Check if we have data with geographic components
    has_geo_data = False
    for col in ['latitude', 'longitude', 'neighborhood', 'zip_code', 'zipcode']:
        if col in combined_data.columns:
            has_geo_data = True
            break
    
    if has_geo_data:
        # Create columns for the layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Location-Based Value Insights")
            
            # Try to create a geographic visualization based on available columns
            geo_col = None
            if 'neighborhood' in combined_data.columns:
                geo_col = 'neighborhood'
            elif 'zip_code' in combined_data.columns:
                geo_col = 'zip_code'
            elif 'zipcode' in combined_data.columns:
                geo_col = 'zipcode'
            
            if geo_col:
                # Find metrics to map
                metrics = []
                for metric in ['median_property_price', 'median_airbnb_price', 'airbnb_density', 'rent_to_income_ratio']:
                    if metric in combined_data.columns:
                        metrics.append(metric)
                
                if metrics:
                    # Let user select a metric to view
                    selected_metric = st.selectbox(
                        "Select metric to visualize by location:",
                        metrics,
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                    
                    # Create aggregation by location
                    geo_data = combined_data.groupby(geo_col)[selected_metric].mean().reset_index()
                    geo_data = geo_data.sort_values(selected_metric, ascending=False)
                    
                    fig = px.bar(
                        geo_data.head(10),
                        x=geo_col,
                        y=selected_metric,
                        title=f'Top 10 Areas by {selected_metric.replace("_", " ").title()}',
                        color=selected_metric,
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show location value ratio metrics if we have the data
                    if 'median_property_price' in combined_data.columns and 'median_airbnb_price' in combined_data.columns:
                        st.markdown("### Airbnb Price to Property Value Ratio")
                        st.markdown("""
                            This metric shows the ratio of nightly Airbnb price to property value (in thousands).
                            Higher values indicate areas where short-term rentals may provide better yields relative to property values.
                        """)
                        
                        # Calculate ratio of Airbnb price to property price (in thousands)
                        combined_data['airbnb_to_property_ratio'] = combined_data['median_airbnb_price'] / (combined_data['median_property_price'] / 1000)
                        
                        ratio_by_loc = combined_data.groupby(geo_col)['airbnb_to_property_ratio'].mean().reset_index()
                        ratio_by_loc = ratio_by_loc.sort_values('airbnb_to_property_ratio', ascending=False)
                        
                        fig = px.bar(
                            ratio_by_loc.head(10),
                            x=geo_col,
                            y='airbnb_to_property_ratio',
                            title='Top 10 Areas by Airbnb Yield Potential',
                            color='airbnb_to_property_ratio',
                            color_continuous_scale='Viridis',
                            labels={'airbnb_to_property_ratio': 'Airbnb $ per $1000 Property Value'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Opportunity Areas")
            
            # Identify areas with good investment potential if we have enough data
            if 'airbnb_count' in combined_data.columns and 'median_property_price' in combined_data.columns and geo_col:
                # Create a scoring system for investment opportunity
                # Normalize the metrics to 0-1 scale for fair comparison
                if len(combined_data) > 1: # Need at least 2 data points for min-max scaling
                    # Copy data to avoid modifying the original dataframe
                    opportunity_df = combined_data.copy()
                    
                    # Define metrics that contribute positively to investment opportunity
                    positive_metrics = [
                        'airbnb_count', 'airbnb_density', 'median_airbnb_price'
                    ]
                    available_pos_metrics = [m for m in positive_metrics if m in opportunity_df.columns]
                    
                    # Define metrics where lower values are better for investment
                    negative_metrics = [
                        'median_property_price', 'rent_to_income_ratio'
                    ]
                    available_neg_metrics = [m for m in negative_metrics if m in opportunity_df.columns]
                    
                    # Normalize and score each metric if we have enough metrics
                    if available_pos_metrics or available_neg_metrics:
                        # Initialize opportunity score
                        opportunity_df['opportunity_score'] = 0
                        
                        for metric in available_pos_metrics:
                            min_val = opportunity_df[metric].min()
                            max_val = opportunity_df[metric].max()
                            range_val = max_val - min_val
                            if range_val > 0:  # Avoid division by zero
                                opportunity_df[f'{metric}_score'] = (opportunity_df[metric] - min_val) / range_val
                                opportunity_df['opportunity_score'] += opportunity_df[f'{metric}_score']
                        
                        for metric in available_neg_metrics:
                            min_val = opportunity_df[metric].min()
                            max_val = opportunity_df[metric].max()
                            range_val = max_val - min_val
                            if range_val > 0:  # Avoid division by zero
                                opportunity_df[f'{metric}_score'] = 1 - ((opportunity_df[metric] - min_val) / range_val)
                                opportunity_df['opportunity_score'] += opportunity_df[f'{metric}_score']
                        
                        # Average the score based on number of metrics used
                        total_metrics = len(available_pos_metrics) + len(available_neg_metrics)
                        if total_metrics > 0:
                            opportunity_df['opportunity_score'] = opportunity_df['opportunity_score'] / total_metrics
                        
                        # Aggregate by location and get top opportunities
                        top_opportunities = opportunity_df.groupby(geo_col)['opportunity_score'].mean().reset_index()
                        top_opportunities = top_opportunities.sort_values('opportunity_score', ascending=False)
                        
                        # Create visualization
                        fig = px.bar(
                            top_opportunities.head(10),
                            x=geo_col,
                            y='opportunity_score',
                            title='Top 10 Investment Opportunity Areas',
                            color='opportunity_score',
                            color_continuous_scale='Viridis',
                            labels={'opportunity_score': 'Opportunity Score (0-1)'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show explanation of the opportunity score
                        with st.expander("How is the Opportunity Score calculated?"):
                            st.markdown(f"""                    
                                The Opportunity Score combines multiple factors to identify areas with strong investment potential:
                                
                                **Positive factors** (higher is better):
                                {', '.join([m.replace('_', ' ').title() for m in available_pos_metrics])}
                                
                                **Negative factors** (lower is better):
                                {', '.join([m.replace('_', ' ').title() for m in available_neg_metrics])}
                                
                                Each factor is normalized to a 0-1 scale and averaged. A higher score suggests better investment potential.
                            """)
                        
                        # Show top 5 areas with details
                        st.markdown("### Top 5 Areas: Key Metrics")
                        top5_areas = top_opportunities.head(5)[geo_col].tolist()
                        
                        # Make sure to only include numeric columns for calculation
                        if len(available_pos_metrics) > 0 or len(available_neg_metrics) > 0:
                            # Filter only to the relevant areas
                            filtered_areas = opportunity_df[opportunity_df[geo_col].isin(top5_areas)]
                            
                            # Create empty dataframe to hold results
                            area_details = pd.DataFrame()
                            area_details[geo_col] = filtered_areas[geo_col].unique()
                            
                            # Calculate means for each metric separately to avoid type errors
                            for metric in available_pos_metrics + available_neg_metrics:
                                try:
                                    # Only perform mean on numeric columns
                                    if pd.api.types.is_numeric_dtype(filtered_areas[metric]):
                                        means = filtered_areas.groupby(geo_col)[metric].mean()
                                        # Add this metric to the results
                                        for area in area_details[geo_col]:
                                            if area in means.index:
                                                area_details.loc[area_details[geo_col] == area, metric] = means[area]
                                except Exception as e:
                                    st.warning(f"Could not calculate mean for {metric}: {str(e)}")
                        else:
                            # If no metrics available, just show the areas
                            area_details = pd.DataFrame({geo_col: top5_areas})
                        
                        # Format the table nicely
                        format_dict = {}
                        for col in area_details.columns:
                            if 'price' in col:
                                format_dict[col] = '${:,.0f}'
                            elif 'ratio' in col:
                                format_dict[col] = '{:.2f}'
                            elif 'count' in col or 'density' in col:
                                format_dict[col] = '{:.1f}'
                        
                        if format_dict:
                            st.dataframe(area_details.style.format(format_dict))
                        else:
                            st.dataframe(area_details)
            
            # If no opportunity analysis available, show a basic map or other geo visualization
            elif 'latitude' in combined_data.columns and 'longitude' in combined_data.columns:
                st.markdown("### Geographic Distribution")
                
                # Sample the data to avoid overplotting
                map_data = combined_data.sample(min(len(combined_data), 1000))
                
                # Find a numeric column to use for colors
                color_col = None
                for col in ['median_property_price', 'median_airbnb_price', 'airbnb_density']:
                    if col in map_data.columns:
                        color_col = col
                        break
                
                if color_col:
                    fig = px.scatter_mapbox(
                        map_data,
                        lat='latitude',
                        lon='longitude',
                        color=color_col,
                        size_max=15,
                        zoom=10,
                        mapbox_style="open-street-map",
                        title=f"{color_col.replace('_', ' ').title()} by Location"
                    )
                else:
                    fig = px.scatter_mapbox(
                        map_data,
                        lat='latitude',
                        lon='longitude',
                        size_max=15,
                        zoom=10,
                        mapbox_style="open-street-map",
                        title="Property Locations"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Geographic data is not available. Please ensure the dataset contains location information.")
        
        # Show tabular summary instead if available
        if 'neighborhood' in combined_data.columns or 'zip_code' in combined_data.columns or 'zipcode' in combined_data.columns:
            geo_col = next(col for col in ['neighborhood', 'zip_code', 'zipcode'] if col in combined_data.columns)
            
            metrics = [col for col in combined_data.columns if col != geo_col and combined_data[col].dtype in [np.float64, np.int64]]
            
            if metrics and geo_col:
                top_areas = combined_data.groupby(geo_col)[metrics].mean().reset_index()
                st.dataframe(top_areas)

def show_correlation_analysis(data_loader, filtered_data):
    """
    Show correlation analysis between key metrics using columns from miami_dade_merged_data.csv.
    
    Args:
        data_loader: DataLoader instance
        filtered_data: Filtered combined dataset
    """
    st.subheader("Correlation Analysis")
    
    if not filtered_data.empty:
        # Identify numerical columns for correlation analysis
        numerical_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out irrelevant columns if present
        exclude_cols = ['index', 'id', 'zip_code']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if len(numerical_cols) > 1:  # Need at least 2 columns for correlation
            corr_data = filtered_data[numerical_cols].corr().round(2)
            
            # Create heatmap
            fig = px.imshow(
                corr_data,
                text_auto=True,
                color_continuous_scale='RdBu_r',  # Blue to Red, reversed
                title='Correlation Matrix',
                labels={'color': 'Correlation'}
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight key findings
            st.markdown("### Key Correlation Findings")
            
            # Get top 3 correlations (excluding self-correlations)
            corrs = []
            for i, col1 in enumerate(corr_data.columns):
                for j, col2 in enumerate(corr_data.columns):
                    if i < j:  # Only take upper triangle, excluding diagonal
                        corrs.append((col1, col2, abs(corr_data.loc[col1, col2])))
            
            # Sort by correlation strength
            corrs = sorted(corrs, key=lambda x: x[2], reverse=True)
            
            # Display top correlations
            for i, (col1, col2, corr_val) in enumerate(corrs[:3]):
                st.markdown(f"- **{col1.replace('_', ' ').title()} vs {col2.replace('_', ' ').title()}**: "
                          f"{corr_data.loc[col1, col2]:.2f} correlation")
        else:
            st.info("Insufficient data for correlation analysis.")
    else:
        # Load image if available
        try:
            st.image("Presentation/images/airbnb_correlation_chart.png", 
                    caption="Correlation between Airbnb Density and Housing Variables")
        except:
            st.info("Correlation analysis visualization not available.")
