#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Home Page Module

This module contains the code for the application's home page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def show_home_page(agent):
    """
    Display the home page content.
    
    This page provides an overview of the application, key findings,
    and instructions for navigating to other sections.
    
    Args:
        agent: HousingImpactAgent instance for proactive insights
    """
    # Import DataLoader within function to avoid circular imports
    # Create main sections for home page
    st.header("Miami Housing Impact Hub")
    st.subheader("Understanding the Impact of Short-Term Rentals on Housing Affordability")
    
    st.markdown("""
        This application provides interactive tools to analyze how short-term rentals
        like Airbnb affect housing affordability across Miami-Dade County neighborhoods.
    """)
    
    # Access the DataLoader instance from session state to check data availability
    if 'data_loader' in st.session_state:
        data_loader = st.session_state['data_loader']
        if data_loader.combined_data is not None and not data_loader.combined_data.empty:
            # Use real data for insights
            display_data_driven_insights(data_loader)    
        st.markdown("""
            Our analysis combines data from multiple sources:
            * Census Bureau American Community Survey (ACS)
            * Miami-Dade County Housing Data
            * Airbnb Listings Data
            
            Through this application, you can explore interactive visualizations,
            predict affordability impacts, and understand the potential consequences
            of different rental regulation approaches.
        """)
    
def display_data_driven_insights(data_loader):
    """Display insights driven by the actual data."""
    # Key Findings Section
    st.markdown("## Key Findings")
    
    # Create columns for the key findings
    col1, col2 = st.columns(2)
    
    # Get the actual data
    data = data_loader.combined_data
    
    with col1:
        st.markdown("### Impact on Affordability")
        
        # Calculate average property price and airbnb price if available
        if 'median_property_price' in data.columns and 'median_airbnb_price' in data.columns:
            avg_property = data['median_property_price'].mean()
            avg_airbnb = data['median_airbnb_price'].mean()
            
            st.markdown(f"""
                Analysis of Miami-Dade County data reveals an average property price of **${avg_property:,.0f}**
                while the average Airbnb price is **${avg_airbnb:.0f}** per night. This pricing dynamic
                creates incentives for property owners to convert long-term housing to short-term rentals.
            """)
        else:
            st.markdown("""
                Analysis of Miami-Dade County data reveals that neighborhoods with high Airbnb 
                density typically show significantly different housing market characteristics,
                indicating potential impacts on affordability for long-term residents.
            """)
        
        try:
            st.image("Presentation/images/affordability_impact.png", 
                    caption="Housing Market Impact by Airbnb Density")
        except:
            # If no image, display data-based visualization if available
            if 'median_property_price' in data.columns and 'airbnb_density' in data.columns:
                fig = px.scatter(
                    data, 
                    x='airbnb_density', 
                    y='median_property_price',
                    hover_name='neighborhood',
                    title='Property Prices vs Airbnb Density',
                    color='airbnb_density',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Geographic Concentration")
        
        # Find top neighborhoods by airbnb count/density if available
        if 'neighborhood' in data.columns and 'airbnb_count' in data.columns:
            top_area = data.sort_values('airbnb_count', ascending=False).iloc[0]
            st.markdown(f"""
                Short-term rentals in Miami-Dade are heavily concentrated in specific areas,
                with **{top_area['neighborhood']}** having the highest concentration at 
                **{top_area['airbnb_count']}** Airbnb listings.
            """)
        else:
            st.markdown("""
                Short-term rentals in Miami-Dade are heavily concentrated in tourist areas
                and coastal neighborhoods, with some areas having significantly higher densities
                of listings compared to the county average.
            """)
        
        try:
            st.image("Presentation/images/geographic_concentration.png", 
                    caption="Airbnb Listing Density by Neighborhood")
        except:
            # If no image, display data-based visualization if available
            if 'neighborhood' in data.columns and 'airbnb_count' in data.columns:
                # Create bar chart of top neighborhoods by airbnb count
                top_n = min(len(data), 5)  # Get top 5 or fewer if less data available
                top_areas = data.sort_values('airbnb_count', ascending=False).head(top_n)
                
                fig = px.bar(
                    top_areas,
                    x='neighborhood',
                    y='airbnb_count',
                    title=f'Top {top_n} Neighborhoods by Airbnb Listings',
                    color='airbnb_count',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)

    # Key Insights section based on real data
    st.subheader("Key Insights")
    
    # Generate insights directly from the data
    insights = []
    
    # Only generate insights if we have data
    if data is not None and not data.empty:
        # Insight 1: Top Airbnb areas if we have that data
        if 'neighborhood' in data.columns and 'airbnb_count' in data.columns:
            top_areas = data.sort_values('airbnb_count', ascending=False).head(3)
            areas_text = ", ".join([f"{row['neighborhood']} ({row['airbnb_count']} listings)" 
                                  for _, row in top_areas.iterrows()])
            
            insights.append({
                'title': 'Airbnb Concentration Hotspots',
                'description': f"The areas with the highest number of Airbnb listings are {areas_text}."
            })
        
        # Insight 2: Affordability data if available
        affordability_col = None
        for col in ['rent_to_income_ratio', 'affordability_index', 'rent_burden']:
            if col in data.columns:
                affordability_col = col
                break
                
        if affordability_col:
            avg_affordability = data[affordability_col].mean()
            high_burden_count = len(data[data[affordability_col] > 30])
            high_burden_pct = (high_burden_count / len(data)) * 100
            
            insights.append({
                'title': 'Housing Affordability Crisis',
                'description': f"The average {affordability_col.replace('_', ' ')} in Miami-Dade is {avg_affordability:.1f}%. Approximately {high_burden_pct:.1f}% of analyzed areas exceed the 30% affordability threshold."
            })
        
        # Insight 3: Any correlation between Airbnb and prices/rents if we have the data
        if 'airbnb_density' in data.columns and 'median_property_price' in data.columns:
            # Calculate correlation
            corr = data['airbnb_density'].corr(data['median_property_price'])
            direction = "positive" if corr > 0 else "negative"
            
            insights.append({
                'title': 'Airbnb and Property Prices',
                'description': f"There is a {direction} correlation ({corr:.2f}) between Airbnb density and property prices, suggesting that {('areas with more Airbnbs tend to have higher property values' if corr > 0 else 'Airbnb presence may not be driving up property values in all areas')}."
            })
    
    # If we don't have enough insights from data, add generic ones
    if len(insights) < 3:
        generic_insights = [
            {
                'title': 'Short-Term Rental Impact',
                'description': "Short-term rentals may reduce the housing supply available for long-term residents, potentially contributing to housing affordability challenges in popular tourist areas."
            },
            {
                'title': 'Neighborhood Transformation',
                'description': "Areas with high Airbnb concentrations often experience changes in neighborhood character, local business mix, and may face community concerns about noise and safety."
            },
            {
                'title': 'Economic Opportunities',
                'description': "Despite affordability concerns, short-term rentals provide income opportunities for property owners and may support tourism-related businesses in the surrounding neighborhoods."
            }
        ]
        
        # Add generic insights until we have at least 3
        for insight in generic_insights:
            if len(insights) >= 3:
                break
            if not any(i['title'] == insight['title'] for i in insights):
                insights.append(insight)
    
    # Display insights in an engaging format
    cols = st.columns(min(3, len(insights)))
    for i, insight in enumerate(insights[:3]):  # Show up to 3 insights
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {insight['title']}")
                st.markdown(insight['description'])
    
    # Get project data overview
    if 'data_loader' in st.session_state:
        data_loader = st.session_state['data_loader']
        if data_loader.combined_data is not None and not data_loader.combined_data.empty:
            # Show data overview
            with st.expander("Data Overview"):
                st.markdown("### Miami-Dade Housing and Airbnb Dataset")
                
                data = data_loader.combined_data
                st.markdown(f"**Total Records:** {len(data)} neighborhoods/areas")
                
                # List available columns with descriptions
                st.markdown("**Available Data Fields:**")
                col_descriptions = {
                    'neighborhood': 'Name of the neighborhood or area',
                    'property_count': 'Number of properties in the area',
                    'airbnb_count': 'Number of Airbnb listings',
                    'airbnb_density': 'Ratio of Airbnb listings to properties',
                    'median_property_price': 'Median price of properties ($)',
                    'median_airbnb_price': 'Median nightly price of Airbnb listings ($)',
                    'population': 'Total population in the area',
                    'median_income': 'Median household income ($)'
                }
                
                # Create a dataframe of column descriptions for available columns
                available_cols = [col for col in col_descriptions if col in data.columns]
                if available_cols:
                    col_df = pd.DataFrame({
                        'Field': available_cols,
                        'Description': [col_descriptions[col] for col in available_cols]
                    })
                    st.table(col_df)
                    
                    # Show sample data
                    st.markdown("**Sample Data:**")
                    st.dataframe(data.head(3))

    # Navigation guidance
    st.subheader("Explore the Application")
    st.markdown("""
        Use the sidebar navigation to explore different aspects of the analysis:
        
        * **Interactive Dashboard**: Explore neighborhood-level data with interactive maps and charts
        * **Affordability Prediction**: Predict how changes in short-term rental density might affect affordability
        * **Housing Assistant**: Get personalized insights and answers to your questions
        * **About the Project**: Learn more about the methodology and data sources
    """)
    
    # Application Sections Overview
    st.markdown("## Explore the Application")
    
    # Display application sections with descriptions
    with st.container(border=True):
        st.markdown("### Key Features")
        
        feature_descriptions = [
            {
                "title": "Interactive Dashboard",
                "description": "Visualize Airbnb distribution and affordability metrics across Miami-Dade neighborhoods.",
                "page": "Interactive Dashboard"
            },
            {
                "title": "Affordability Prediction",
                "description": "Analyze how changes in short-term rental density might affect housing affordability.",
                "page": "Affordability Prediction"
            },
            {
                "title": "Housing Assistant",
                "description": "Get answers to questions about housing affordability and short-term rental impacts.",
                "page": "Housing Assistant"
            }
        ]
        
        # Create a three-column layout for features
        cols = st.columns(len(feature_descriptions))
        
        for i, (col, feature) in enumerate(zip(cols, feature_descriptions)):
            with col:
                st.markdown(f"**{feature['title']}**")
                st.markdown(feature['description'])
                if st.button(f"Open {feature['title']}", key=f"feature_{i}"):
                    st.session_state['page'] = feature['page']
                    st.rerun()

    # Recommended explorations based on available data
    st.subheader("Recommended Explorations")
    
    # Create data-driven recommendations
    recommendations = [
        {
            'title': 'Interactive Dashboard',
            'description': 'Explore detailed maps and charts showing the distribution of Airbnb listings and housing affordability metrics across Miami-Dade neighborhoods.',
            'page': 'Dashboard'
        },
        {
            'title': 'Affordability Predictions',
            'description': 'Analyze how potential changes in short-term rental density might impact housing affordability in specific neighborhoods.',
            'page': 'Affordability Prediction'
        },
        {
            'title': 'Ask the Assistant',
            'description': 'Get answers to specific questions about the relationship between short-term rentals and housing affordability in Miami-Dade County.',
            'page': 'Housing Assistant'
        }
    ]
    
    # Display recommendations with buttons
    rec_cols = st.columns(min(3, len(recommendations)))
    for i, recommendation in enumerate(recommendations[:3]):  # Show up to 3 recommendations
        with rec_cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {recommendation['title']}")
                st.markdown(recommendation['description'])
                
                if st.button(f"Explore", key=f"home_rec_{i}"):
                    # Set the page to the recommended page
                    st.session_state['page'] = recommendation['page']
                    
                    # Force a rerun to navigate to the recommended page
                    st.rerun()
    
    # Call to action
    st.info("""
        **Ask the Housing Assistant** to get personalized insights and answers 
        about the impact of short-term rentals on housing affordability in Miami-Dade County.
    """)
