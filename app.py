
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------
# LOAD DATA AND MODEL
# --------------------------------

df = pd.read_csv("real_estate_processed.csv.gz", compression="gzip")
model = joblib.load("investment_model.pkl")

# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------

st.sidebar.title("🏠 Real Estate Advisor")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Property Analysis",
        "Find Properties",
        "Dashboard"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Real Estate Investment Advisor")


# ================================================
# 1. HOME PAGE
# ================================================

if page == "Home":

    st.title("🏠 Real Estate Investment Advisor")

    st.subheader(
        "Predicting Property Profitability & Future Value"
    )

    st.write(
        "Welcome to the Real Estate Investment Advisor. "
        "This application helps users explore properties, "
        "analyze investment indicators and estimate future prices."
    )

    st.divider()

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Properties",
        f"{len(df):,}"
    )

    col2.metric(
        "Total Cities",
        df["City"].nunique()
    )

    col3.metric(
        "Property Types",
        df["Property_Type"].nunique()
    )

    st.divider()

    st.subheader("Available Features")

    st.markdown("""
    - **Property Analysis:** Enter property details and analyze investment indicators.
    - **Future Price Estimation:** Estimate property value after five years.
    - **Find Properties:** Search and filter properties.
    - **Dashboard:** Explore property market insights through charts.
    """)


# ================================================
# 2. PROPERTY ANALYSIS PAGE
# ================================================

elif page == "Property Analysis":

    st.title("📝 Property Analysis")

    st.write("Enter the property details below.")

    st.header("Property Details")

    city = st.selectbox(
        "Select City",
        sorted(df["City"].dropna().unique())
    )

    bhk = st.selectbox(
        "Select BHK",
        sorted(df["BHK"].dropna().unique())
    )

    property_type = st.selectbox(
        "Select Property Type",
        sorted(df["Property_Type"].dropna().unique())
    )

    size = st.number_input(
        "Property Size (SqFt)",
        min_value=100,
        max_value=10000,
        value=1200
    )

    price = st.number_input(
        "Property Price (Lakhs)",
        min_value=1.0,
        max_value=10000.0,
        value=50.0
    )

    furnished = st.selectbox(
        "Furnished Status",
        sorted(df["Furnished_Status"].dropna().unique())
    )

    st.header("Additional Property Details")

    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2026,
        value=2020
    )

    age = 2026 - year_built

    nearby_schools = st.number_input(
        "Nearby Schools",
        min_value=0,
        max_value=20,
        value=2
    )

    nearby_hospitals = st.number_input(
        "Nearby Hospitals",
        min_value=0,
        max_value=20,
        value=2
    )

    floor_no = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=100,
        value=1
    )

    total_floors = st.number_input(
        "Total Floors",
        min_value=1,
        max_value=100,
        value=5
    )

    st.divider()

    if st.button("Analyze Property", type="primary"):

        st.success("Property Details Submitted!")

        st.subheader("Submitted Property Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("City:", city)
            st.write("BHK:", bhk)
            st.write("Property Type:", property_type)

        with col2:
            st.write("Property Size:", size, "SqFt")
            st.write("Property Price:", price, "Lakhs")
            st.write("Furnished Status:", furnished)

        # MODEL INPUT DATA

        input_data = pd.DataFrame([{
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Price_in_Lakhs": price,
            "Price_per_SqFt": price / size,
            "Year_Built": year_built,
            "Age_of_Property": age,
            "Nearby_Schools": nearby_schools,
            "Nearby_Hospitals": nearby_hospitals,
            "Floor_No": floor_no,
            "Total_Floors": total_floors
        }])

        # CLASSIFICATION PREDICTION

        # ================= CLASSIFICATION PREDICTION =================

        prediction = model.predict(input_data)

        st.divider()
        st.header("Investment Analysis")

        if prediction[0] == 1:
            st.success(
                "Good Investment - Based on Dataset Rule"
            )
        else:
            st.warning(
                "Not Good Investment - Based on Dataset Rule"
            )

        # Show the rule-based checks for transparency
        median_price = df["Price_in_Lakhs"].median()
        median_price_sqft = df["Price_per_SqFt"].median()

        st.subheader("Investment Rule Check")

        price_check = price <= median_price
        sqft_check = (price / size) <= median_price_sqft

        col1, col2 = st.columns(2)

        with col1:
            st.write("Price below dataset median:")
            if price_check:
                st.success("Yes")
            else:
                st.warning("No")

        with col2:
            st.write("Price per SqFt below dataset median:")
            if sqft_check:
                st.success("Yes")
            else:
                st.warning("No")

        st.caption(
            "Classification follows the dataset rule: "
            "both price and price per SqFt must be at or below "
            "their respective dataset medians."
        )

        # ================= FUTURE PRICE ESTIMATION =================

        st.divider()

        st.header("5-Year Future Price Estimation")

        growth_rate = 0.08
        years = 5

        future_price = price * (1 + growth_rate) ** years
        expected_increase = future_price - price

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Current Price",
            f"₹{price:.2f} Lakhs"
        )

        col2.metric(
            "Estimated Future Price",
            f"₹{future_price:.2f} Lakhs"
        )

        col3.metric(
            "Expected Increase",
            f"₹{expected_increase:.2f} Lakhs"
        )

        st.info(
            "Future price is calculated using an assumed "
            "8% annual growth rate for 5 years. "
            "This is an illustrative estimate, not a guaranteed prediction."
        )

        st.warning(
            "The investment classification is based on a "
            "dataset-derived rule. It does not measure actual "
            "investment success or guarantee returns."
        )

        
# ================================================
# 3. FIND PROPERTIES PAGE
# ================================================

elif page == "Find Properties":

    st.title("🔍 Find Properties")

    st.write("Search properties using different filters.")

    st.header("Filter Properties")

    filter_city = st.selectbox(
        "Select City",
        ["All"] + sorted(df["City"].dropna().unique())
    )

    filter_bhk = st.selectbox(
        "Select BHK",
        ["All"] + sorted(df["BHK"].dropna().unique())
    )

    filter_type = st.selectbox(
        "Select Property Type",
        ["All"] + sorted(df["Property_Type"].dropna().unique())
    )

    min_price = float(df["Price_in_Lakhs"].min())

    max_price = float(df["Price_in_Lakhs"].max())

    price_range = st.slider(
        "Select Price Range (Lakhs)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )

    # APPLY FILTERS

    filtered_df = df[
        (df["Price_in_Lakhs"] >= price_range[0]) &
        (df["Price_in_Lakhs"] <= price_range[1])
    ]

    if filter_city != "All":

        filtered_df = filtered_df[
            filtered_df["City"] == filter_city
        ]

    if filter_bhk != "All":

        filtered_df = filtered_df[
            filtered_df["BHK"] == filter_bhk
        ]

    if filter_type != "All":

        filtered_df = filtered_df[
            filtered_df["Property_Type"] == filter_type
        ]

    st.divider()

    st.subheader("Filtered Properties")

    st.metric(
        "Total Properties Found",
        f"{len(filtered_df):,}"
    )

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True
    )


# ================================================
# 4. DASHBOARD PAGE
# ================================================


elif page == "Dashboard":

    # ================= DASHBOARD THEME =================
    st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
    }

    section[data-testid="stSidebar"] {
        background-color: #102A43;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    h1, h2, h3 {
        color: #102A43 !important;
    }

    /* KPI Cards */
    div[data-testid="stMetric"] {
        background-color: #163A5F !important;
        border: 1px solid #244F73 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        min-height: 90px;
    }

    /* KPI Headings */
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"] span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: bold !important;
    }

    /* KPI Values */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 23px !important;
        font-weight: bold !important;
    }

    /* Chart containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #B8CCE0 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        padding: 10px !important;
    }
    /* Force KPI headings to white */
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] span {
    color: white !important;
    -webkit-text-fill-color: white !important;
    opacity: 1 !important;
}
    </style>
    """, unsafe_allow_html=True)

    st.title("Real Estate Investment Dashboard")
    st.caption("Interactive Property Market Analysis")

    # ================= SIDEBAR FILTERS =================
    st.sidebar.header("Dashboard Filters")

    # City dropdown
    city_options = sorted(df["City"].dropna().unique().tolist())

    selected_city = st.sidebar.multiselect(
        "Select City",
        options=city_options,
        default=city_options,
        placeholder="Choose City"
    )

    # Locality dropdown
    locality_options = sorted(
        df[df["City"].isin(selected_city)]["Locality"]
        .dropna().unique().tolist()
    )

    selected_locality = st.sidebar.multiselect(
        "Select Locality",
        options=locality_options,
        default=locality_options,
        placeholder="Choose Locality"
    )

    # BHK dropdown
    bhk_options = sorted(df["BHK"].dropna().unique().tolist())

    selected_bhk = st.sidebar.multiselect(
        "Select BHK",
        options=bhk_options,
        default=bhk_options,
        placeholder="Choose BHK"
    )

    # Property Type dropdown
    property_options = sorted(
        df["Property_Type"].dropna().unique().tolist()
    )

    selected_property = st.sidebar.multiselect(
        "Property Type",
        options=property_options,
        default=property_options,
        placeholder="Choose Property Type"
    )

    # Furnished Status dropdown
    furnished_options = sorted(
        df["Furnished_Status"].dropna().unique().tolist()
    )

    selected_furnished = st.sidebar.multiselect(
        "Furnished Status",
        options=furnished_options,
        default=furnished_options,
        placeholder="Choose Furnished Status"
    )

    # Price Range
    min_price = float(df["Price_in_Lakhs"].min())
    max_price = float(df["Price_in_Lakhs"].max())

    selected_price = st.sidebar.slider(
        "Price Range (Lakhs)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )

    # ================= FILTER DATA =================
    dashboard_df = df[
        (df["City"].isin(selected_city)) &
        (df["Locality"].isin(selected_locality)) &
        (df["BHK"].isin(selected_bhk)) &
        (df["Property_Type"].isin(selected_property)) &
        (df["Furnished_Status"].isin(selected_furnished)) &
        (df["Price_in_Lakhs"].between(
            selected_price[0],
            selected_price[1]
        ))
    ].copy()

    # ================= DISPLAY DASHBOARD =================
    if dashboard_df.empty:

        st.warning(
            "No properties found. Please change your filters."
        )

    else:

        # ================= KPI CARDS =================
        st.subheader("Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Properties",
                f"{len(dashboard_df):,}"
            )

        with col2:
            st.metric(
                "Average Price",
                f"₹ {dashboard_df['Price_in_Lakhs'].mean():,.2f} L"
            )

        with col3:
            st.metric(
                "Average Size",
                f"{dashboard_df['Size_in_SqFt'].mean():,.0f} SqFt"
            )

        with col4:
            st.metric(
                "Average BHK",
                f"{dashboard_df['BHK'].mean():.1f}"
            )

        st.divider()

        # ================= CHART ROW 1 =================
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):

                st.subheader("City-wise Average Price")

                city_price = (
                    dashboard_df.groupby("City")["Price_in_Lakhs"]
                    .mean()
                    .reset_index()
                    .sort_values(
                        "Price_in_Lakhs",
                        ascending=False
                    )
                )

                fig = px.bar(
                    city_price,
                    x="City",
                    y="Price_in_Lakhs",
                    color="Price_in_Lakhs",
                    color_continuous_scale=[
                        "#DCEAF7",
                        "#8FB9DD",
                        "#356B9A",
                        "#102A43"
                    ],
                    labels={
                        "Price_in_Lakhs": "Average Price (Lakhs)"
                    }
                )

                fig.update_layout(
                    template="plotly_white",
                    height=350,
                    xaxis_title="City",
                    yaxis_title="Average Price (Lakhs)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with col2:
            with st.container(border=True):

                st.subheader("Property Type-wise Average Price")

                type_price = (
                    dashboard_df.groupby("Property_Type")[
                        "Price_in_Lakhs"
                    ]
                    .mean()
                    .reset_index()
                )

                fig = px.bar(
                    type_price,
                    x="Property_Type",
                    y="Price_in_Lakhs",
                    color="Property_Type",
                    color_discrete_sequence=[
                        "#102A43",
                        "#356B9A",
                        "#8FB9DD",
                        "#DCEAF7"
                    ],
                    labels={
                        "Price_in_Lakhs": "Average Price (Lakhs)"
                    }
                )

                fig.update_layout(
                    template="plotly_white",
                    height=350,
                    showlegend=False,
                    xaxis_title="Property Type",
                    yaxis_title="Average Price (Lakhs)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ================= CHART ROW 2 =================
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):

                st.subheader("Top 10 Localities by Average Price")

                locality_price = (
                    dashboard_df.groupby("Locality")[
                        "Price_in_Lakhs"
                    ]
                    .mean()
                    .reset_index()
                    .sort_values(
                        "Price_in_Lakhs",
                        ascending=False
                    )
                    .head(10)
                )

                fig = px.bar(
                    locality_price,
                    x="Price_in_Lakhs",
                    y="Locality",
                    orientation="h",
                    color="Price_in_Lakhs",
                    color_continuous_scale=[
                        "#DCEAF7",
                        "#8FB9DD",
                        "#356B9A",
                        "#102A43"
                    ]
                )

                fig.update_layout(
                    template="plotly_white",
                    height=400,
                    yaxis={"categoryorder": "total ascending"},
                    xaxis_title="Average Price (Lakhs)",
                    yaxis_title="Locality"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with col2:
            with st.container(border=True):

                st.subheader("Property Price Distribution")

                fig = px.histogram(
                    dashboard_df,
                    x="Price_in_Lakhs",
                    nbins=30,
                    color_discrete_sequence=["#356B9A"],
                    labels={
                        "Price_in_Lakhs": "Price (Lakhs)"
                    }
                )

                fig.update_layout(
                    template="plotly_white",
                    height=400,
                    xaxis_title="Property Price (Lakhs)",
                    yaxis_title="Number of Properties"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ================= CHART ROW 3 =================
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):

                st.subheader("Investment Classification")

                if "Good_Investment" in dashboard_df.columns:

                    investment_counts = (
                        dashboard_df["Good_Investment"]
                        .map({
                            1: "Good Investment",
                            0: "Not Good Investment"
                        })
                        .value_counts()
                        .rename_axis("Investment Status")
                        .reset_index(name="Count")
                    )

                    fig = px.pie(
                        investment_counts,
                        names="Investment Status",
                        values="Count",
                        hole=0.45,
                        color_discrete_sequence=[
                            "#102A43",
                            "#8FB9DD"
                        ]
                    )

                    fig.update_layout(
                        template="plotly_white",
                        height=380
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:
                    st.info(
                        "Good_Investment column is not available."
                    )

        with col2:
            with st.container(border=True):

                st.subheader("City vs Property Type Price")

                heatmap_data = dashboard_df.pivot_table(
                    values="Price_in_Lakhs",
                    index="City",
                    columns="Property_Type",
                    aggfunc="mean"
                )

                if not heatmap_data.empty:

                    fig = px.imshow(
                        heatmap_data,
                        color_continuous_scale=[
                            "#DCEAF7",
                            "#8FB9DD",
                            "#356B9A",
                            "#102A43"
                        ],
                        labels={
                            "color": "Average Price (Lakhs)"
                        },
                        aspect="auto"
                    )

                    fig.update_layout(
                        template="plotly_white",
                        height=380
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:
                    st.info("No data available for heatmap.")

        # ================= FEATURE IMPORTANCE =================
        st.divider()

        with st.container(border=True):

            st.subheader("Machine Learning Feature Importance")

            if hasattr(model, "feature_importances_"):

                importance_df = pd.DataFrame({
                    "Feature": model.feature_names_in_,
                    "Importance": model.feature_importances_
                }).sort_values(
                    "Importance",
                    ascending=True
                )

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=[
                        "#DCEAF7",
                        "#8FB9DD",
                        "#356B9A",
                        "#102A43"
                    ]
                )

                fig.update_layout(
                    template="plotly_white",
                    height=450,
                    xaxis_title="Feature Importance",
                    yaxis_title="Feature"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info(
                    "Feature importance is not available for this model."
                )

        # ================= DISCLAIMER =================
        st.caption(
            "Note: Investment classification is based on a derived "
            "rule-based label. It does not guarantee actual investment "
            "returns. Feature importance shows the model's learned "
            "feature contributions, not causal impact."
        )
