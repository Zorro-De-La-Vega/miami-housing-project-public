#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Miami Housing Impact Hub - Main Application

This is the main entry point for the Streamlit application that visualizes
the impact of short-term rentals on housing affordability in Miami-Dade County.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add the project root to the Python path so we can import modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import application modules
from src.app.utils.data_loader import DataLoader
from src.app.utils.agent import HousingImpactAgent
from src.app.utils.summary_logger import SummaryLogger
from src.app.pages.home import show_home_page
from src.app.pages.dashboard import show_dashboard_page
from src.app.pages.prediction import show_prediction_page
from src.app.pages.about import show_about_page
from src.app.pages.assistant import show_assistant_page
from src.app.pages.advanced_analysis import show_advanced_analysis_page

# Configure the Streamlit page
st.set_page_config(
    page_title="Miami Housing Impact Hub",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize application state in session state if not already present
if 'data_loader' not in st.session_state:
    st.session_state['data_loader'] = DataLoader(project_root)
    
# Initialize the summary logger
if 'summary_logger' not in st.session_state:
    st.session_state['summary_logger'] = SummaryLogger(project_root)
    
# Initialize the agent with data loader and summary logger
if 'agent' not in st.session_state:
    st.session_state['agent'] = HousingImpactAgent(
        st.session_state['data_loader'],
        st.session_state['summary_logger']
    )
    
# Initialize user interaction tracking
if 'page_views' not in st.session_state:
    st.session_state['page_views'] = {}

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Application title and description
st.title("Miami Housing Impact Hub")
st.markdown("""
    ### Analyzing the Impact of Short-Term Rentals on Housing Affordability in Miami-Dade County
    
    This application provides interactive visualizations and predictive analytics to understand 
    how short-term rentals affect housing affordability across Miami-Dade neighborhoods.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", 
    ["Home", "Interactive Dashboard", "Advanced Analysis", "Affordability Prediction", "Housing Assistant", "About the Project"])

# Track page views for the agent
if page not in st.session_state['page_views']:
    st.session_state['page_views'][page] = 0
st.session_state['page_views'][page] += 1

# Log page view and update agent with page interaction
st.session_state['summary_logger'].log_interaction(page, 'page_view')
st.session_state['agent'].update_from_user_interaction({'page': page})

# Display the selected page
if page == "Home":
    show_home_page(st.session_state['agent'])
elif page == "Interactive Dashboard":
    show_dashboard_page(st.session_state['data_loader'], st.session_state['agent'])
elif page == "Advanced Analysis":
    show_advanced_analysis_page(st.session_state['data_loader'], st.session_state['agent'])
elif page == "Affordability Prediction":
    show_prediction_page(st.session_state['data_loader'], st.session_state['agent'])
elif page == "Housing Assistant":
    show_assistant_page(st.session_state['agent'])
elif page == "About the Project":
    show_about_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Sources")
st.sidebar.markdown("""
- Census Bureau American Community Survey
- Miami-Dade County Housing Data
- Airbnb Listings Data
""")

st.sidebar.markdown("### Project Information")
st.sidebar.markdown("""
Miami Housing Impact Hub v0.1.0  
Data Mining Final Project  
© 2025
""")

# Add option to generate summary statistics
st.sidebar.markdown("---")
if st.sidebar.button("Generate Summary Reports"):
    with st.sidebar.status("Generating summary statistics...", expanded=True) as status:
        st.sidebar.write("Processing application logs...")
        st.session_state['summary_logger'].generate_summary_stats()
        st.sidebar.write("Exporting agent insights...")
        export_file = st.session_state['summary_logger'].export_insights_to_csv()
        status.update(label="Summary statistics generated!", state="complete", expanded=True)
        
        if export_file:
            st.sidebar.success(f"Key insights exported to {export_file.name}")

if __name__ == "__main__":
    # This part only executes when the script is run directly, not when imported
    pass
