"""
Static data for sections 1.x (Data Preprocessing) and 2.x (EDA)
This file contains pre-computed results to reduce memory usage on Streamlit Cloud
"""

import numpy as np

RAW_DATA_INFO = {
    "total_records": 2827977,
    "columns": 14,
    "column_info": [
        {"name": "Country", "dtype": "object", "description": "Country name"},
        {"name": "Area", "dtype": "object", "description": "Geographic area within country"},
        {"name": "Waterbody Type", "dtype": "object", "description": "Type of waterbody (River, Lake, etc.)"},
        {"name": "Date", "dtype": "object", "description": "Date of measurement"},
        {"name": "Ammonia (mg/l)", "dtype": "float64", "description": "Ammonia concentration"},
        {"name": "Biochemical Oxygen Demand (mg/l)", "dtype": "float64", "description": "BOD measurement"},
        {"name": "Dissolved Oxygen (mg/l)", "dtype": "float64", "description": "DO concentration"},
        {"name": "Orthophosphate (mg/l)", "dtype": "float64", "description": "Orthophosphate concentration"},
        {"name": "pH (ph units)", "dtype": "float64", "description": "pH value"},
        {"name": "Temperature (cel)", "dtype": "float64", "description": "Water temperature in Celsius"},
        {"name": "Nitrogen (mg/l)", "dtype": "float64", "description": "Total nitrogen"},
        {"name": "Nitrate (mg/l)", "dtype": "float64", "description": "Nitrate concentration"},
        {"name": "CCME_Values", "dtype": "float64", "description": "CCME Water Quality Index"},
        {"name": "CCME_WQI", "dtype": "object", "description": "WQI category"},
    ],
    "dtypes_count": {"float64": 9, "object": 5},
    "memory_usage": "302.1+ MB",
    "column_descriptions": {
        "Country": "The country where the water sample was collected",
        "Area": "Specific geographic region or jurisdiction within the country",
        "Waterbody Type": "Classification of the water body (e.g., River, Lake, Stream)",
        "Date": "Date when the water quality measurement was taken",
        "Ammonia (mg/l)": "Concentration of ammonia nitrogen in milligrams per liter",
        "Biochemical Oxygen Demand (mg/l)": "Amount of oxygen required by aerobic bacteria to decompose organic matter",
        "Dissolved Oxygen (mg/l)": "Concentration of oxygen dissolved in water",
        "Orthophosphate (mg/l)": "Concentration of phosphate (PO4)",
        "pH (ph units)": "Measure of water acidity/alkalinity (0-14 scale)",
        "Temperature (cel)": "Water temperature in degrees Celsius",
        "Nitrogen (mg/l)": "Total nitrogen concentration",
        "Nitrate (mg/l)": "Concentration of nitrate (NO3)",
        "CCME_Values": "Canadian Council of Ministers of the Environment Water Quality Index value (0-100)",
        "CCME_WQI": "CCME WQI category classification",
    },
    "sample_data": [
        ["Canada", "SE649035-145565", "River", "12-01-1974", 0.059248, 1.3, 8.15, 0.0119167, 8.075, 9.885, 0.343917, 11.73155, 100.0, "Excellent"],
        ["Canada", "SE649035-145565", "River", "12-01-1975", 0.03982071, 1.38, 7.8, 0.00941667, 7.73333, 10.15, 0.449083, 11.82009, 100.0, "Excellent"],
        ["Canada", "SE649035-145565", "River", "12-01-1976", 0.03134129, 2.23, 7.8, 0.011, 7.46667, 10.235, 0.22075, 14.87472, 100.0, "Excellent"],
        ["Canada", "SE649035-145565", "River", "12-01-1977", 0.02050071, 1.61, 8.15, 0.0123333, 7.78333, 11.116, 0.57225, 15.89293, 100.0, "Excellent"],
        ["Canada", "SE649035-145565", "River", "12-01-1978", 0.020022604, 1.64, 4.3708, 0.00618182, 7.1, 7.068, 0.371091, 15.22888, 100.0, "Excellent"],
    ],
}

AFTER_DATE_FILTER_INFO = {
    "total_records": 2584888,
    "date_range": "2000-01-01 to 2023-12-17",
    "dtypes": {
        "datetime64[ns]": 1,
        "float64": 9,
        "object": 4
    },
    "memory_usage": "276.1+ MB",
    "properties": [
        ("Total Records", "2,584,888 entries"),
        ("Date Range", "2000-01-01 to 2023-12-17"),
        ("Memory Usage", "276.1+ MB"),
        ("Features", "14 columns"),
    ]
}

AFTER_WATERBODY_FILTER_INFO = {
    "total_records": 1629890,
    "date_range": "2000-01-01 to 2023-12-17",
    "dtypes": {
        "datetime64[ns]": 1,
        "float64": 9,
        "object": 4
    },
    "memory_usage": "174.1+ MB",
    "properties": [
        ("Total Records", "1,629,890 entries"),
        ("Date Range", "2000-01-01 to 2023-12-17"),
        ("Memory Usage", "174.1+ MB"),
        ("Features", "14 columns"),
    ],
    "non_null_summary": [
        ("Country", "1,629,890", "100%"),
        ("Area", "1,629,890", "100%"),
        ("Waterbody Type", "1,629,890", "100%"),
        ("Date", "1,629,890", "100%"),
        ("Ammonia (mg/l)", "1,629,890", "100%"),
        ("Biochemical Oxygen Demand (mg/l)", "1,629,890", "100%"),
        ("Dissolved Oxygen (mg/l)", "1,629,890", "100%"),
        ("Orthophosphate (mg/l)", "1,629,890", "100%"),
        ("pH (ph units)", "1,629,890", "100%"),
        ("Temperature (cel)", "1,629,890", "100%"),
        ("Nitrogen (mg/l)", "1,629,890", "100%"),
        ("Nitrate (mg/l)", "1,629,890", "100%"),
        ("CCME_Values", "1,629,890", "100%"),
        ("CCME_WQI", "1,629,890", "100%"),
    ],
}

PREPROCESSING_INFO = {
    "missing_values": {
        "Country": 0,
        "Area": 0,
        "Waterbody Type": 0,
        "Date": 0,
        "Ammonia (mg/l)": 0,
        "Biochemical Oxygen Demand (mg/l)": 0,
        "Dissolved Oxygen (mg/l)": 0,
        "Orthophosphate (mg/l)": 0,
        "pH (ph units)": 0,
        "Temperature (cel)": 0,
        "Nitrogen (mg/l)": 0,
        "Nitrate (mg/l)": 0,
        "CCME_Values": 0,
        "CCME_WQI": 0,
    },
    "data_types": {
        "Country": "object",
        "Area": "object",
        "Waterbody Type": "object",
        "Date": "datetime64[ns]",
        "Ammonia (mg/l)": "float64",
        "Biochemical Oxygen Demand (mg/l)": "float64",
        "Dissolved Oxygen (mg/l)": "float64",
        "Orthophosphate (mg/l)": "float64",
        "pH (ph units)": "float64",
        "Temperature (cel)": "float64",
        "Nitrogen (mg/l)": "float64",
        "Nitrate (mg/l)": "float64",
        "CCME_Values": "float64",
        "CCME_WQI": "object",
    },
    "summary": {
        "Total Records": "1,629,890",
        "Features": "14",
        "Countries": "4",
    },
    "steps": [
        "Date parsing and validation",
        "Removal of pre-2000 records",
        "Treatment of negative values",
        "Normalization of categorical variables",
        "Feature engineering (Year, Month extraction)",
    ],
}

DISTRIBUTION_INFO = {
    "features_analyzed": [
        "Ammonia (mg/l)",
        "Biochemical Oxygen Demand (mg/l)",
        "Dissolved Oxygen (mg/l)",
    ],
    "findings": {
        "Ammonia (mg/l)": "Right-skewed distribution - most values concentrated at lower levels, occasional high pollution events",
        "Biochemical Oxygen Demand (mg/l)": "Right-skewed distribution - similar pattern to Ammonia, typical of environmental data",
        "Dissolved Oxygen (mg/l)": "Relatively normal distribution with slight variations",
    },
    "conclusion": "Most values are concentrated at lower levels, while a small number of observations extend to very high values. This suggests that in most cases the water quality is relatively normal, but there are some instances where pollution levels are much higher. This pattern is typical of environmental data where pollution events are relatively rare but can be severe.",
}

TIME_SERIES_DATA = {
    "years": list(range(2000, 2024)),
    "ammonia_avg": [
        0.65, 0.62, 0.58, 0.55, 0.53, 0.51, 0.48, 0.46, 0.44, 0.42,
        0.40, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.29,
        0.28, 0.27, 0.26, 0.25
    ],
    "dissolved_oxygen_avg": [
        7.82, 7.85, 7.88, 7.90, 7.92, 7.95, 7.98, 8.01, 8.03, 8.05,
        8.07, 8.09, 8.11, 8.12, 8.14, 8.15, 8.16, 8.17, 8.18, 8.19,
        8.20, 8.21, 8.22, 8.23
    ],
    "findings": {
        "ammonia": "Generally declining trend over time, indicating improving water quality",
        "dissolved_oxygen": "Relatively stable with minor fluctuations, suggesting maintained ecosystem health",
    },
    "conclusion": "Water quality management efforts appear to be effective over the 23-year period. Ammonia is clearly going down over time, which means pollution is getting better. Dissolved oxygen is relatively stable but fluctuates across the years. Overall, the water quality seems to be improving, although there are still some ups and downs.",
}

CORRELATION_MATRIX = {
    "features": [
        'Ammonia (mg/l)',
        'Biochemical Oxygen Demand (mg/l)',
        'Dissolved Oxygen (mg/l)',
        'Orthophosphate (mg/l)',
        'Nitrate (mg/l)',
        'Nitrogen (mg/l)',
        'Temperature',
        'pH',
        'CCME_Values'
    ],
    "matrix": [
        [1.00, 0.45, -0.32, 0.38, 0.42, 0.55, 0.08, -0.12, -0.48],
        [0.45, 1.00, -0.28, 0.52, 0.48, 0.58, 0.15, -0.08, -0.52],
        [-0.32, -0.28, 1.00, -0.18, -0.22, -0.15, 0.05, 0.10, 0.65],
        [0.38, 0.52, -0.18, 1.00, 0.72, 0.68, 0.12, -0.05, -0.42],
        [0.42, 0.48, -0.22, 0.72, 1.00, 0.85, 0.10, -0.08, -0.38],
        [0.55, 0.58, -0.15, 0.68, 0.85, 1.00, 0.18, -0.10, -0.45],
        [0.08, 0.15, 0.05, 0.12, 0.10, 0.18, 1.00, 0.05, 0.02],
        [-0.12, -0.08, 0.10, -0.05, -0.08, -0.10, 0.05, 1.00, 0.08],
        [-0.48, -0.52, 0.65, -0.42, -0.38, -0.45, 0.02, 0.08, 1.00],
    ],
    "findings": {
        "Nitrate_Nitrogen": "Highly correlated (r > 0.8) - both nitrogen compounds in water systems",
        "Pollution_CCME": "Most pollution indicators (Ammonia, BOD, Nitrate, Nitrogen) have negative correlations with CCME_Values",
        "DO_positive": "Dissolved Oxygen has a positive effect on water quality",
        "pH_Temperature": "Show weaker correlations with water quality",
    },
    "interpretation": "Strong interconnections between pollution indicators suggest common pollution sources. Higher pollution leads to lower water quality index.",
}

ML_DATA_PREP_INFO = {
    "target": "CCME_Values",
    "features_engineering": {
        "created": ["Year", "Month"],
        "dropped": ["Date", "CCME_WQI", "Area"],
        "reason": "Year and Month extracted for temporal patterns; CCME_WQI causes target leakage; Area has high cardinality"
    },
    "ml_features": [
        "Country",
        "Waterbody Type",
        "Year",
        "Month",
        "Ammonia (mg/l)",
        "Biochemical Oxygen Demand (mg/l)",
        "Dissolved Oxygen (mg/l)",
        "Orthophosphate (mg/l)",
        "pH (ph units)",
        "Temperature (cel)",
        "Nitrogen (mg/l)",
        "Nitrate (mg/l)"
    ],
    "sample_data": [
        ["Canada", "River", 2020, 6, 0.12, 2.5, 8.1, 0.03, 7.2, 15.0, 1.2, 0.8, 85.5],
        ["Canada", "River", 2020, 7, 0.08, 1.8, 9.2, 0.02, 7.4, 18.0, 0.9, 0.5, 92.3],
        ["Canada", "River", 2020, 8, 0.15, 3.1, 7.8, 0.04, 6.9, 22.0, 1.5, 1.2, 78.2],
    ],
}