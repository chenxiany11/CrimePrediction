sybilQAQ
sybilqaq
Online

sybilQAQ — 11/20/24, 22:50
？
four pears — 11/20/24, 22:50
哈哈哈哈
也不怪 就是那个package
他就是会操作的时候
create a new instance
无语
sybilQAQ — 11/20/24, 22:50
无语
four pears — 11/20/24, 22:50
干脆弄成feature。。
浏览一下～
sybilQAQ — 11/20/24, 22:54
我们要显示这些信息吗 （3mile改成1mile
Image
还是就不要high low
Image
four pears — 11/20/24, 22:55
可以显示！！
多多益善
他屁事多
sybilQAQ — 11/20/24, 22:55
确实
全显示上
sybilQAQ — 11/20/24, 23:10
Attachment file type: unknown
data_preprocessing.ipynb
8.20 KB
这个是改完的
给你的小八看看
four pears — 11/20/24, 23:11
小八是不是看不懂ipybn
nb
sybilQAQ — 11/20/24, 23:11
复制给他
就两段
four pears — 11/20/24, 23:11
ok
哈哈
你怎么知道我偷懒
呜呜
sybilQAQ — 11/20/24, 23:12
哈哈哈哈哈哈
因为我也偷懒
four pears — 11/21/24, 00:41
dy
dy
dy
dy
dy
dy
dy
dy
dy
dy
dy
dy
dy
dy
dy
sybilQAQ — 11/21/24, 04:46
dldldldl
sybilQAQ — 11/21/24, 18:32
docker file
four pears — 11/22/24, 00:06
宝宝宝宝
生日快乐
sybilQAQ — 11/22/24, 00:06
宝宝宝宝
谢谢宝宝
sybilQAQ started a call. — Today at 14:56
sybilQAQ — Today at 15:20
Image
sybilQAQ — Today at 15:33
offense_start_datetime,precinct,offense_parent_group,latitude,longitude,crime_against_category
2020-02-05 10:10:00,W,DRUG/NARCOTIC OFFENSES,47.64938723,-122.385973723,SOCIETY
2020-02-03 08:00:00,N,LARCENY-THEFT,47.67511789,-122.323399063,PROPERTY
2020-02-02 20:30:00,N,ROBBERY,47.66638407,-122.29955218,PROPERTY
2020-02-05 01:17:00,W,DESTRUCTION/DAMAGE/VANDALISM OF PROPERTY,47.64292734,-122.384864805,PROPERTY
2020-02-05 00:51:21,N,DRIVING UNDER THE INFLUENCE,47.66219308,-122.366195342,SOCIETY... (32 MB left)
Expand
filtered_dataset_2019_to_now.csv
33 MB
four pears — Today at 15:40
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import os

RANDOM_STATE = 42

st.set_page_config(page_title="Crime Data Analysis and Prediction", layout="wide")
st.title("📊 Crime Data Analysis and Prediction App")

# ----------------------------
# 1. Data Loading Function
# ----------------------------

@st.cache_data(ttl=3600)
def load_local_data(file_path):
    """
    Load data from a local CSV file.

    Args:
        file_path (str): Path to the local CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        st.success("Data loaded successfully from the local dataset.")
        return df
    except FileNotFoundError:
        st.error(f"File not found at path: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

# ----------------------------
# 2. Model Loading Function
# ----------------------------

@st.cache_resource
def load_models():
    """
    Load pre-trained models and label encoders.

    Returns:
        tuple: classifier_risk, classifier_crime, le_crime, le_risk
    """
    try:
        classifier_risk = joblib.load('models/classifier_risk.pkl')
        classifier_crime = joblib.load('models/classifier_crime.pkl')
        le_crime = joblib.load('models/le_crime.pkl')
        le_risk = joblib.load('models/le_risk.pkl')
        st.success("Models and encoders loaded successfully.")
        return classifier_risk, classifier_crime, le_crime, le_risk
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# ----------------------------
# 3. Data Preprocessing Function
# ----------------------------

@st.cache_data(ttl=3600)
def preprocess_data(df):
    """
    Preprocess the DataFrame:
    - Clean column names
    - Filter necessary columns
    - Handle missing values
    - Encode categorical variables
    - Feature engineering

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        tuple: Preprocessed DataFrame, le_crime, le_risk
    """
    # Clean column names: strip spaces, convert to lowercase, replace spaces and slashes with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    
    # Define necessary columns (adjusted for capitalized attribute names in the original dataset)
    columns_needed = [
        'Offense_Start_Datetime',
        'Precinct',
        'Offense_Parent_Group',
        'Latitude',
        'Longitude',
        'Crime_Against_Category',
    ]
    
    # Adjust column names to lowercase with underscores
... (274 lines left)
Collapse
message.txt
14 KB
streamlit run crimeapp.py
﻿
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import os

RANDOM_STATE = 42

st.set_page_config(page_title="Crime Data Analysis and Prediction", layout="wide")
st.title("📊 Crime Data Analysis and Prediction App")

# ----------------------------
# 1. Data Loading Function
# ----------------------------

@st.cache_data(ttl=3600)
def load_local_data(file_path):
    """
    Load data from a local CSV file.

    Args:
        file_path (str): Path to the local CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        st.success("Data loaded successfully from the local dataset.")
        return df
    except FileNotFoundError:
        st.error(f"File not found at path: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

# ----------------------------
# 2. Model Loading Function
# ----------------------------

@st.cache_resource
def load_models():
    """
    Load pre-trained models and label encoders.

    Returns:
        tuple: classifier_risk, classifier_crime, le_crime, le_risk
    """
    try:
        classifier_risk = joblib.load('models/classifier_risk.pkl')
        classifier_crime = joblib.load('models/classifier_crime.pkl')
        le_crime = joblib.load('models/le_crime.pkl')
        le_risk = joblib.load('models/le_risk.pkl')
        st.success("Models and encoders loaded successfully.")
        return classifier_risk, classifier_crime, le_crime, le_risk
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# ----------------------------
# 3. Data Preprocessing Function
# ----------------------------

@st.cache_data(ttl=3600)
def preprocess_data(df):
    """
    Preprocess the DataFrame:
    - Clean column names
    - Filter necessary columns
    - Handle missing values
    - Encode categorical variables
    - Feature engineering

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        tuple: Preprocessed DataFrame, le_crime, le_risk
    """
    # Clean column names: strip spaces, convert to lowercase, replace spaces and slashes with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    
    # Define necessary columns (adjusted for capitalized attribute names in the original dataset)
    columns_needed = [
        'Offense_Start_Datetime',
        'Precinct',
        'Offense_Parent_Group',
        'Latitude',
        'Longitude',
        'Crime_Against_Category',
    ]
    
    # Adjust column names to lowercase with underscores
    columns_needed = [col.lower().replace(' ', '_').replace('/', '_') for col in columns_needed]
    
    # Check if all necessary columns are present
    missing_cols = set(columns_needed) - set(df.columns)
    if missing_cols:
        st.error(f"Missing columns in the dataset: {missing_cols}")
        return pd.DataFrame(), None, None
    
    # Select and copy necessary columns
    df = df[columns_needed].copy()
    
    # Drop rows with any missing values
    df.dropna(inplace=True)
    
    # Convert 'offense_start_datetime' to datetime
    df['offense_start_datetime'] = pd.to_datetime(df['offense_start_datetime'], errors='coerce')
    df.dropna(subset=['offense_start_datetime'], inplace=True)
    
    # Standardize 'offense_parent_group' by stripping and converting to uppercase
    df['offense_parent_group'] = df['offense_parent_group'].str.strip().str.upper()
    
    # Filter data for the most recent year
    recent_year = df['offense_start_datetime'].max() - pd.DateOffset(years=1)
    df = df[df['offense_start_datetime'] >= recent_year]
    
    # Remove rare offenses (those appearing once or less)
    offense_counts = df['offense_parent_group'].value_counts()
    rare_offenses = offense_counts[offense_counts <= 1].index.tolist()
    if rare_offenses:
        df = df[~df['offense_parent_group'].isin(rare_offenses)]

    # Define high-risk categories
    high_risk_categories = [
        'ASSAULT OFFENSES', 'HOMICIDE OFFENSES', 'KIDNAPPING/ABDUCTION', 'ROBBERY',
        'WEAPON LAW VIOLATIONS', 'BURGLARY/BREAKING&ENTERING', 'MOTOR VEHICLE THEFT',
        'DESTRUCTION/DAMAGE/VANDALISM OF PROPERTY', 'DRUG/NARCOTIC OFFENSES',
        'EXTORTION/BLACKMAIL', 'HUMAN TRAFFICKING'
    ]
    
    # Create 'risk_level' based on high-risk categories
    df['risk_level'] = np.where(df['offense_parent_group'].isin(high_risk_categories), 'High', 'Low')
    
    # Initialize label encoders
    le_crime = LabelEncoder()
    le_risk = LabelEncoder()
    
    # Fit and transform 'offense_parent_group' and 'risk_level'
    le_crime.fit(df['offense_parent_group'])
    df['crime_type'] = le_crime.transform(df['offense_parent_group'])
    
    le_risk.fit(df['risk_level'])
    df['risk_level_encoded'] = le_risk.transform(df['risk_level'])
    
    # Feature engineering: extract month, day of week, and hour from datetime
    df['month'] = df['offense_start_datetime'].dt.month
    df['day_of_week'] = df['offense_start_datetime'].dt.dayofweek
    df['hour'] = df['offense_start_datetime'].dt.hour
    
    return df, le_crime, le_risk

# ----------------------------
# 4. Feature Preparation Function
# ----------------------------

@st.cache_data(ttl=3600)
def prepare_features(df):
    """
    Prepare features for model prediction:
    - Select relevant features
    - Apply one-hot encoding

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        pd.DataFrame: Feature matrix.
    """
    X = df[['month', 'day_of_week', 'hour', 'latitude', 'longitude']]
    X = pd.get_dummies(X, columns=['month', 'day_of_week', 'hour'], drop_first=True)
    return X

# ----------------------------
# 5. Main Execution
# ----------------------------

with st.spinner('Loading and processing data...'):
    # Define the path to your local dataset
    local_data_path = './filtered_dataset_2019_to_now.csv'  # Update this path as needed
    
    # Check if the file exists
    if not os.path.exists(local_data_path):
        st.error(f"The dataset file was not found at the path: {local_data_path}")
        st.stop()
    
    # Load data
    df_raw = load_local_data(local_data_path)
    if df_raw.empty:
        st.warning("No data loaded. Please check the file path or the dataset content.")
        st.stop()
    
    # Preprocess data
    df, le_crime, le_risk = preprocess_data(df_raw)
    if df.empty:
        st.warning("Preprocessed data is empty. Please check the dataset and preprocessing steps.")
        st.stop()
    
    # Prepare features
    X = prepare_features(df)
    y_risk = df['risk_level_encoded']
    y_crime = df['crime_type']
    
    # Load models
    classifier_risk, classifier_crime, le_crime_loaded, le_risk_loaded = load_models()
    if not all([classifier_risk, classifier_crime, le_crime_loaded, le_risk_loaded]):
        st.error("One or more models or encoders failed to load. Please check the logs.")
        st.stop()

# ----------------------------
# 6. Sidebar for Filters
# ----------------------------

st.sidebar.header("🔍 Filter and View Crime Data")
st.sidebar.subheader("Select Time Range")

# Define date range
min_date = df['offense_start_datetime'].min().date()
max_date = df['offense_start_datetime'].max().date()

# Date inputs
start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

# Validate date inputs
if start_date > end_date:
    st.sidebar.error("**Start Date** must be before **End Date**.")

# Crime type selection
st.sidebar.subheader("Select Crime Types")
crime_types = sorted(df['offense_parent_group'].unique())
selected_crime_types = st.sidebar.multiselect("Crime Types", crime_types, default=crime_types)

# Default map center (Seattle)
user_lat = 47.6062
user_lon = -122.3321

# ----------------------------
# 7. Apply Filters
# ----------------------------

mask = (
    (df['offense_start_datetime'].dt.date >= start_date) &
    (df['offense_start_datetime'].dt.date <= end_date) &
    (df['offense_parent_group'].isin(selected_crime_types))
)
filtered_df = df[mask].copy()

# ----------------------------
# 8. Map Visualization
# ----------------------------

st.subheader("📍 Crime Incidents Map")
st.markdown(f"📈 Displaying data from **{start_date}** to **{end_date}**")
st.markdown("""**Interact with the Map**: Click on any location on the map to view a heatmap of crimes within a 1-mile radius and see detailed risk assessments.""")

# Initialize Folium map
m = folium.Map(location=[user_lat, user_lon], zoom_start=13)
marker_cluster = MarkerCluster().add_to(m)

# Add CircleMarkers to the map
for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='red' if row['risk_level'] == 'High' else 'blue',
        fill=True,
        fill_color='red' if row['risk_level'] == 'High' else 'blue',
        fill_opacity=0.6,
        popup=f"{row['offense_parent_group']} on {row['offense_start_datetime'].date()}"
    ).add_to(marker_cluster)

# Display the map
map_data = st_folium(m, width=700, height=500)

# ----------------------------
# 9. Interactive Map Features
# ----------------------------

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    clicked_location = (clicked["lat"], clicked["lng"])
    st.write(f"**Clicked Location:** Latitude {clicked_location[0]:.4f}, Longitude {clicked_location[1]:.4f}")
    
    # Calculate distance from clicked location to all incidents
    df['distance_to_click'] = df.apply(
        lambda row: geodesic(clicked_location, (row['latitude'], row['longitude'])).miles,
        axis=1
    )
    
    # Filter incidents within 1 mile
    nearby_df = df[df['distance_to_click'] <= 1].copy()
    
    if not nearby_df.empty:
        # Prepare heatmap data
        heat_data = nearby_df[['latitude', 'longitude']].values.tolist()
        
        # Initialize heatmap
        heat_map = folium.Map(location=clicked_location, zoom_start=14)
        HeatMap(heat_data, radius=15, blur=10).add_to(heat_map)
        
        # Add a circle to represent 1-mile radius
        folium.Circle(
            location=clicked_location,
            radius=1609,  # 1 mile in meters
            color='green',
            fill=False
        ).add_to(heat_map)
        
        # Display heatmap
        st.subheader("🔥 Heatmap of Crimes within 1 Mile")
        st_folium(heat_map, width=700, height=500)
        
        # Compute risk levels
        total_incidents = len(nearby_df)
        high_risk_count = (nearby_df['risk_level'] == 'High').sum()
        low_risk_count = total_incidents - high_risk_count
        risk_level = 'High' if high_risk_count / total_incidents >= 0.5 else 'Low'
        
        # Display risk assessment
        st.markdown(f"### **Risk Level for the Area:** {risk_level}")
        st.markdown(f"- **Total incidents within 1-mile radius:** {total_incidents}")
        st.markdown(f"- **High Risk incidents:** {high_risk_count} ({high_risk_count / total_incidents:.2%})")
        st.markdown(f"- **Low Risk incidents:** {low_risk_count} ({low_risk_count / total_incidents:.2%})")
        
        # Risk Level Explanations
        st.markdown("""
        **Risk Level Explanations:**
        - **High Risk**: Indicates that **50% or more** of the incidents in the area are categorized as high-risk crimes (e.g., Assault Offenses, Robbery). Such areas may require increased vigilance and safety measures.
        - **Low Risk**: Indicates that **less than 50%** of the incidents are high-risk crimes. These areas are generally considered safer, but it's always important to stay informed.
        """)
        
        # Prepare features for nearby incidents
        X_nearby = nearby_df[['month', 'day_of_week', 'hour', 'latitude', 'longitude']].copy()
        X_nearby = pd.get_dummies(X_nearby, columns=['month', 'day_of_week', 'hour'], drop_first=True)
        X_nearby = X_nearby.reindex(columns=prepare_features(df).columns, fill_value=0)
        
        # Predict crime type probabilities
        proba_crime = classifier_crime.predict_proba(X_nearby)
        avg_proba_crime = proba_crime.mean(axis=0)
        crime_proba_sorted = sorted(zip(le_crime.classes_, avg_proba_crime), key=lambda x: x[1], reverse=True)
        
        # Display top 5 crime type probabilities
        st.markdown("### **Top 5 Crime Type Probabilities:**")
        for crime, prob in crime_proba_sorted[:5]:
            st.markdown(f"- **{crime}**: {prob:.2%}")
    else:
        st.write("No incidents found within a 1-mile radius.")

# ----------------------------
# 10. Overall Crime Analysis
# ----------------------------

st.subheader("🔍 Overall Crime Analysis")

# Risk Level Distribution
risk_distribution = filtered_df['risk_level'].value_counts().reset_index()
risk_distribution.columns = ['Risk Level', 'Count']
st.markdown("#### 📊 Risk Level Distribution")
st.bar_chart(risk_distribution.set_index('Risk Level'))

# Crime Types Distribution
crime_distribution = filtered_df['offense_parent_group'].value_counts().reset_index()
crime_distribution.columns = ['Crime Type', 'Count']
st.markdown("#### 📊 Crime Types Distribution")
st.bar_chart(crime_distribution.set_index('Crime Type'))
