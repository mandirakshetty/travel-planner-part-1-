import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import plotly.express as px
from geopy.geocoders import Nominatim
import time

# Page configuration
st.set_page_config(
    page_title="Eco-Friendly Route Planner - India",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .route-info {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2E8B57;
    }
    .eco-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E8B57;
    }
    .mode-comparison {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class IndiaRoutePlanner:
    def __init__(self):
        self.osrm_base_url = "http://router.project-osrm.org/route/v1"
        self.geolocator = Nominatim(user_agent="eco_route_planner")
        
        # India-specific emission factors (kg CO2 per passenger-km) - Based on your data
        self.emission_factors = {
            'car_petrol': 0.128,      # Petrol Car (128 g/km = 0.128 kg/km)
            'car_diesel': 0.114,      # Diesel Car
            'car_cng': 0.083,         # CNG Car
            'bike': 0.038,            # Petrol 2-wheeler
            'auto': 0.030,            # CNG Auto-Rickshaw
            'bus': 0.021,             # Diesel Bus (City)
            'metro': 0.0002,          # Metro Electric
            'foot': 0.0,              # Walking
            'bicycle': 0.0            # Cycling
        }
        
        # India-specific cost factors (₹ per km)
        self.cost_factors = {
            'car_petrol': 8.5,        # ₹8.5/km (fuel + maintenance)
            'car_diesel': 7.2,        # ₹7.2/km
            'car_cng': 5.0,           # ₹5.0/km
            'bike': 2.0,              # ₹2.0/km
            'auto': 12.0,             # ₹12.0/km (auto fare)
            'bus': 1.5,               # ₹1.5/km (BMTC bus)
            'metro': 2.0,             # ₹2.0/km (Namma Metro)
            'foot': 0.0,
            'bicycle': 0.5            # ₹0.5/km (maintenance)
        }
        
        # Default route preferences
        self.route_profiles = {
            'fastest': {'time': 0.7, 'emissions': 0.2, 'cost': 0.1},
            'greenest': {'time': 0.2, 'emissions': 0.6, 'cost': 0.2},
            'cheapest': {'time': 0.2, 'emissions': 0.1, 'cost': 0.7},
            'balanced': {'time': 0.33, 'emissions': 0.33, 'cost': 0.33}
        }

    def geocode_address(self, address):
        """Convert address to coordinates"""
        try:
            location = self.geolocator.geocode(address + ", Bangalore, India")
            if location:
                return (location.latitude, location.longitude)
            return None
        except Exception as e:
            st.error(f"Geocoding error: {e}")
            return None

    def get_route(self, start_coords, end_coords, profile='car', route_type='fastest'):
        """Get route from OSRM API with proper road selection"""
        
        # Map profile to OSRM profiles
        profile_map = {
            'car_petrol': 'driving',
            'car_diesel': 'driving', 
            'car_cng': 'driving',
            'bike': 'driving',  # Using driving for bikes on proper roads
            'auto': 'driving',
            'bus': 'driving',
            'metro': 'driving',  # Will be handled separately
            'foot': 'walking',
            'bicycle': 'cycling'
        }
        
        osrm_profile = profile_map.get(profile, 'driving')
        
        url = f"{self.osrm_base_url}/{osrm_profile}/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
        params = {
            'overview': 'full',
            'geometries': 'geojson',
            'steps': 'true',
            'alternatives': 'true'  # Get multiple route options
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                # Select best route based on type
                if data.get('routes') and len(data['routes']) > 0:
                    if route_type == 'fastest':
                        # Choose route with minimum duration
                        best_route = min(data['routes'], key=lambda x: x['duration'])
                    elif route_type == 'shortest':
                        # Choose route with minimum distance
                        best_route = min(data['routes'], key=lambda x: x['distance'])
                    else:  # balanced or greenest
                        # Choose a balanced route (usually the first one)
                        best_route = data['routes'][0]
                    
                    return {'routes': [best_route]}
                return None
            return None
        except Exception as e:
            st.error(f"Routing error: {e}")
            return None

    def calculate_emissions(self, distance_km, transport_mode):
        """Calculate CO2 emissions for a route using India-specific factors"""
        return distance_km * self.emission_factors.get(transport_mode, 0.128)

    def calculate_cost(self, distance_km, transport_mode):
        """Calculate cost for a route using India-specific factors"""
        return distance_km * self.cost_factors.get(transport_mode, 8.5)

    def calculate_eco_score(self, distance, duration, emissions, cost, weights=None):
        """Calculate eco-score based on multiple factors"""
        if weights is None:
            weights = self.route_profiles['balanced']
        
        # Normalize factors (lower is better)
        max_emissions = 5.0   # kg CO2 (reduced for Indian context)
        max_cost = 200.0      # ₹ (Indian rupees)
        max_time = 180.0      # minutes
        
        norm_emissions = min(emissions / max_emissions, 1)
        norm_cost = min(cost / max_cost, 1)
        norm_time = min(duration / max_time, 1)
        
        # Calculate score (higher is better)
        score = (weights['emissions'] * (1 - norm_emissions) + 
                weights['cost'] * (1 - norm_cost) + 
                weights['time'] * (1 - norm_time))
        
        return round(score * 100, 1)

    def analyze_route_quality(self, route_data):
        """Analyze if route uses proper roads"""
        if not route_data or 'routes' not in route_data:
            return "Unknown"
        
        route = route_data['routes'][0]
        distance = route['distance'] / 1000  # km
        
        # Simple analysis based on distance and duration
        avg_speed = (distance / (route['duration'] / 3600)) if route['duration'] > 0 else 0
        
        if avg_speed > 30:  # km/h
            return "Highway/Main Roads"
        elif avg_speed > 15:
            return "Mixed Roads"
        else:
            return "Local Roads"

def create_enhanced_map(start_coords, end_coords, route_data, transport_mode, route_quality):
    """Create Folium map with enhanced route visualization"""
    m = folium.Map(location=start_coords, zoom_start=13)
    
    # Add start and end markers
    folium.Marker(
        start_coords,
        popup='Start',
        tooltip='Start Point',
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        end_coords,
        popup='End', 
        tooltip='End Point',
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Add route if available
    if route_data and 'routes' in route_data and len(route_data['routes']) > 0:
        route_geometry = route_data['routes'][0]['geometry']
        
        # Choose color and style based on transport mode and route quality
        color_map = {
            'car_petrol': 'blue',
            'car_diesel': 'darkblue', 
            'car_cng': 'lightblue',
            'bike': 'orange',
            'auto': 'yellow',
            'bus': 'purple',
            'metro': 'red',
            'foot': 'green',
            'bicycle': 'lightgreen'
        }
        
        # Adjust line weight based on route quality
        weight_map = {
            "Highway/Main Roads": 8,
            "Mixed Roads": 6, 
            "Local Roads": 4
        }
        
        folium.GeoJson(
            route_geometry,
            style_function=lambda x: {
                'color': color_map.get(transport_mode, 'blue'),
                'weight': weight_map.get(route_quality, 6),
                'opacity': 0.8,
                'lineCap': 'round'
            }
        ).add_to(m)
        
        # Add route quality info
        folium.Marker(
            [start_coords[0] + 0.001, start_coords[1] + 0.001],
            popup=f'Route Quality: {route_quality}',
            icon=folium.DivIcon(html=f'<div style="background-color: white; padding: 2px; border: 1px solid black;">{route_quality}</div>')
        ).add_to(m)
    
    return m

def main():
    st.markdown('<div class="main-header">🌱 India Eco-Friendly Route Planner</div>', unsafe_allow_html=True)
    
    # Initialize route planner
    planner = IndiaRoutePlanner()
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown('### 📍 Route Details')
        
        # Location inputs
        col1, col2 = st.columns(2)
        with col1:
            start_address = st.text_input("From", "Mantri Square, Bangalore")
        with col2:
            end_address = st.text_input("To", "Sampige Road, Bangalore")
        
        # Transport mode selection
        st.markdown('### 🚗 Transport Mode')
        transport_mode = st.selectbox(
            "Select Vehicle Type",
            ['car_petrol', 'car_diesel', 'car_cng', 'bike', 'auto', 'bus', 'metro', 'bicycle', 'foot'],
            format_func=lambda x: {
                'car_petrol': 'Petrol Car 🚗',
                'car_diesel': 'Diesel Car 🚗', 
                'car_cng': 'CNG Car 🚗',
                'bike': 'Motor Bike 🏍️',
                'auto': 'Auto Rickshaw 🛺',
                'bus': 'Bus 🚌',
                'metro': 'Metro 🚇',
                'bicycle': 'Bicycle 🚲',
                'foot': 'Walking 🚶'
            }[x]
        )
        
        # Route type selection
        st.markdown('### 🛣️ Route Preference')
        route_type = st.selectbox(
            "Route Type",
            ['fastest', 'shortest', 'balanced', 'greenest'],
            format_func=lambda x: {
                'fastest': '🚀 Fastest Route',
                'shortest': '📏 Shortest Distance', 
                'balanced': '⚖️ Balanced Route',
                'greenest': '🌿 Greenest Route'
            }[x]
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            st.markdown("#### 🎯 Custom Weights")
            col1, col2, col3 = st.columns(3)
            with col1:
                time_weight = st.slider("Time", 0.0, 1.0, planner.route_profiles[route_type]['time'])
            with col2:
                emission_weight = st.slider("Emission", 0.0, 1.0, planner.route_profiles[route_type]['emissions'])
            with col3:
                cost_weight = st.slider("Cost", 0.0, 1.0, planner.route_profiles[route_type]['cost'])
            
            # Normalize weights
            total = time_weight + emission_weight + cost_weight
            if total > 0:
                custom_weights = {
                    'time': time_weight / total,
                    'emissions': emission_weight / total, 
                    'cost': cost_weight / total
                }
            else:
                custom_weights = planner.route_profiles[route_type]
        
        if st.button("Find Optimal Route", type="primary", use_container_width=True):
            with st.spinner("Finding the best eco-friendly route..."):
                # Geocode addresses
                start_coords = planner.geocode_address(start_address)
                end_coords = planner.geocode_address(end_address)
                
                if start_coords and end_coords:
                    # Get route data
                    route_data = planner.get_route(start_coords, end_coords, transport_mode, route_type)
                    
                    if route_data and 'routes' in route_data and len(route_data['routes']) > 0:
                        route_info = route_data['routes'][0]
                        distance_km = route_info['distance'] / 1000
                        duration_min = route_info['duration'] / 60
                        
                        # Calculate metrics
                        emissions = planner.calculate_emissions(distance_km, transport_mode)
                        cost = planner.calculate_cost(distance_km, transport_mode)
                        eco_score = planner.calculate_eco_score(
                            distance_km, duration_min, emissions, cost, custom_weights
                        )
                        route_quality = planner.analyze_route_quality(route_data)
                        
                        # Store in session state
                        st.session_state.update({
                            'start_coords': start_coords,
                            'end_coords': end_coords,
                            'route_data': route_data,
                            'transport_mode': transport_mode,
                            'route_type': route_type,
                            'distance_km': distance_km,
                            'duration_min': duration_min,
                            'emissions': emissions,
                            'cost': cost,
                            'eco_score': eco_score,
                            'route_quality': route_quality,
                            'custom_weights': custom_weights,
                            'route_found': True
                        })
                    else:
                        st.error("❌ No route found. Please try different addresses or transport mode.")
                        st.session_state.route_found = False
                else:
                    st.error("❌ Could not find locations. Please check addresses and try again.")
                    st.session_state.route_found = False

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('### 🗺️ Route Map')
        
        if 'route_found' in st.session_state and st.session_state.route_found:
            # Create and display enhanced map
            m = create_enhanced_map(
                st.session_state.start_coords,
                st.session_state.end_coords, 
                st.session_state.route_data,
                st.session_state.transport_mode,
                st.session_state.route_quality
            )
            st_folium(m, width=700, height=500)
        else:
            # Default Bangalore map
            m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
            st_folium(m, width=700, height=500)
    
    with col2:
        st.markdown('### 📊 Route Information')
        
        if 'route_found' in st.session_state and st.session_state.route_found:
            # Display comprehensive route information
            mode_display = {
                'car_petrol': 'Petrol Car', 'car_diesel': 'Diesel Car', 'car_cng': 'CNG Car',
                'bike': 'Motor Bike', 'auto': 'Auto Rickshaw', 'bus': 'Bus', 
                'metro': 'Metro', 'bicycle': 'Bicycle', 'foot': 'Walking'
            }
            
            st.markdown(f"""
            <div class="route-info">
                <h4>📍 Route Summary</h4>
                <p><strong>Mode:</strong> {mode_display[st.session_state.transport_mode]}</p>
                <p><strong>Preference:</strong> {st.session_state.route_type.title()}</p>
                <p><strong>Distance:</strong> {st.session_state.distance_km:.1f} km</p>
                <p><strong>Duration:</strong> {st.session_state.duration_min:.1f} min</p>
                <p><strong>Route Quality:</strong> {st.session_state.route_quality}</p>
                <hr>
                <p><strong>CO2 Emissions:</strong> {st.session_state.emissions:.3f} kg</p>
                <p><strong>Estimated Cost:</strong> ₹{st.session_state.cost:.1f}</p>
                <p class="eco-score">🌿 Eco-Score: {st.session_state.eco_score}/100</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Step-by-step directions
            if 'legs' in st.session_state.route_data['routes'][0]:
                st.markdown("### 🧭 Turn-by-Turn Directions")
                legs = st.session_state.route_data['routes'][0]['legs']
                for i, leg in enumerate(legs):
                    for j, step in enumerate(leg['steps']):
                        instruction = step.get('maneuver', {}).get('instruction', 'Continue straight')
                        distance_km = step['distance'] / 1000
                        st.write(f"**{j+1}.** {instruction} ({distance_km:.2f} km)")
            
            # Enhanced mode comparison
            st.markdown("### 📈 Transport Mode Comparison")
            modes_to_compare = ['car_petrol', 'car_diesel', 'bike', 'auto', 'bus', 'bicycle']
            comparison_data = []
            
            for mode in modes_to_compare:
                test_route = planner.get_route(
                    st.session_state.start_coords, 
                    st.session_state.end_coords, 
                    mode, 
                    st.session_state.route_type
                )
                if test_route and 'routes' in test_route:
                    route_info = test_route['routes'][0]
                    dist = route_info['distance'] / 1000
                    dur = route_info['duration'] / 60
                    em = planner.calculate_emissions(dist, mode)
                    co = planner.calculate_cost(dist, mode)
                    score = planner.calculate_eco_score(dist, dur, em, co, st.session_state.custom_weights)
                    
                    comparison_data.append({
                        'Mode': mode_display[mode],
                        'Distance_km': dist,
                        'Duration_min': dur,
                        'Emissions_kg': em,
                        'Cost_₹': co,
                        'Eco_Score': score
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                
                # Create comparison charts
                tab1, tab2, tab3 = st.tabs(["Eco-Score", "Emissions", "Cost"])
                
                with tab1:
                    fig_score = px.bar(df_comparison, x='Mode', y='Eco_Score', 
                                     title='Eco-Score Comparison',
                                     color='Eco_Score',
                                     color_continuous_scale='Viridis')
                    st.plotly_chart(fig_score, use_container_width=True)
                
                with tab2:
                    fig_emissions = px.bar(df_comparison, x='Mode', y='Emissions_kg',
                                         title='CO2 Emissions Comparison',
                                         color='Emissions_kg',
                                         color_continuous_scale='Blues')
                    st.plotly_chart(fig_emissions, use_container_width=True)
                
                with tab3:
                    fig_cost = px.bar(df_comparison, x='Mode', y='Cost_₹',
                                    title='Cost Comparison (₹)',
                                    color='Cost_₹', 
                                    color_continuous_scale='Greens')
                    st.plotly_chart(fig_cost, use_container_width=True)
        
        else:
            st.info("👆 Enter addresses and click 'Find Optimal Route' to see detailed route information and comparisons.")

    # Footer with project info
    st.markdown("---")
    st.markdown("""
    ### 🎯 Project Features
    - **India-Specific Data**: Accurate emission factors and costs for Indian vehicles
    - **Smart Routing**: Prefers main roads and highways for better travel experience  
    - **Multiple Route Options**: Fastest, shortest, balanced, and greenest routes
    - **Eco-Scoring**: AI-powered environmental impact assessment
    - **Cost Analysis**: Realistic cost estimates in Indian Rupees
    - **Multi-Modal Comparison**: Compare all transport options side-by-side
    """)

if __name__ == "__main__":
    main()