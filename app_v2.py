import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ECO Meal AI",
    page_icon="🍽️",
    layout="wide"
)


# ============================================================
# LOAD DATA + MODELS
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/food_wastage_data.csv")
    return df.drop_duplicates().copy()


@st.cache_resource
def load_waste_model():
    return joblib.load("models/food_waste_model.pkl")


@st.cache_resource
def load_quantity_model():
    return joblib.load("models/food_quantity_model.pkl")


df = load_data()
waste_model = load_waste_model()
quantity_model = load_quantity_model()


# ============================================================
# HEADER
# ============================================================

st.title("🍽️ ECO Meal AI")

st.markdown(
    """
    ### AI-Powered Food Planning, Waste Prediction & Reduction Platform

    Estimate how much food to prepare, predict potential food waste,
    and generate practical actions for reducing avoidable waste.
    """
)

st.divider()


# ============================================================
# TABS
# ============================================================

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🏠 Dashboard",
        "🍽️ Smart Food Planning",
        "🔮 Waste Prediction",
        "📊 Analytics",
        "♻️ Food Recovery",
        "🎯 Waste Reduction Simulator",
        "📈 Impact Dashboard"
    ]
)


# ============================================================
# TAB 0 — DASHBOARD
# ============================================================

with tab0:

    st.header("🏠 ECO Meal AI")

    st.subheader(
        "AI-Powered Food Planning, Waste Prediction, Reduction & Recovery"
    )

    st.write(
        "ECO Meal AI uses machine learning to estimate preparation quantity "
        "from expected guests and event conditions, predict potential food "
        "waste, and support better preparation decisions."
    )

    st.divider()

    st.subheader("🔄 How the AI System Works")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📊 Records Analyzed",
            f"{len(df):,}"
        )

    with c2:
        st.metric(
            "🤖 ML Models",
            "2"
        )

    with c3:
        st.metric(
            "🍽️ Quantity Model",
            "R² = 0.839"
        )

    with c4:
        st.metric(
            "♻️ Waste Model",
            "Random Forest"
        )

    st.divider()

    st.subheader("🚀 Platform Modules")

    m1, m2 = st.columns(2)

    with m1:

        st.markdown(
            """
            ### 🍽️ 1. Smart Food Planning

            Enter the expected guests and event conditions.
            The AI estimates how much food should be prepared.
            """
        )

        st.markdown(
            """
            ### 🔮 2. Waste Prediction

            Estimate potential food waste using the selected
            event and preparation conditions.
            """
        )

        st.markdown(
            """
            ### 🎯 3. Waste Reduction Simulator

            Test different preparation quantities and compare
            model-based waste estimates.
            """
        )

    with m2:

        st.markdown(
            """
            ### 📊 4. Analytics

            Explore food-waste patterns across food types,
            events, preparation methods and quantities.
            """
        )

        st.markdown(
            """
            ### ♻️ 5. Food Recovery

            Identify potential surplus and guide it through
            a proposed safety-verification workflow.
            """
        )

        st.markdown(
            """
            ### 📈 6. Impact Dashboard

            View dataset-based metrics and relationships
            between guests, food quantity and waste.
            """
        )

    st.divider()

    st.subheader("📋 Dataset Snapshot")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Average Guests",
            f"{df['Number of Guests'].mean():.0f}"
        )

    with d2:
        st.metric(
            "Average Food Quantity",
            f"{df['Quantity of Food'].mean():.1f}"
        )

    with d3:
        st.metric(
            "Average Waste",
            f"{df['Wastage Food Amount'].mean():.1f}"
        )

    st.divider()

    st.info(
        "Prototype status: The quantity and waste models are trained "
        "using the available cleaned dataset. The recovery module is "
        "a proposed workflow and does not perform real-time food-safety "
        "or NGO verification."
    )


# ============================================================
# TAB 1 — SMART FOOD PLANNING
# ============================================================

with tab1:

    st.header("🍽️ Smart Food Planning")

    st.write(
        "Tell ECO Meal AI about the expected event. The quantity model "
        "will estimate the preparation quantity automatically, then "
        "the waste model will estimate potential food waste."
    )

    st.info(
        "💡 You do NOT need to enter the food quantity. "
        "ECO Meal AI predicts it automatically."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        plan_food_type = st.selectbox(
            "Type of Food",
            sorted(df["Type of Food"].unique()),
            key="plan_food_type"
        )

        plan_guests = st.number_input(
            "Number of Guests",
            min_value=int(df["Number of Guests"].min()),
            max_value=int(df["Number of Guests"].max()),
            value=350,
            key="plan_guests"
        )

        plan_event = st.selectbox(
            "Event Type",
            sorted(df["Event Type"].unique()),
            key="plan_event"
        )

        plan_storage = st.selectbox(
            "Storage Conditions",
            sorted(df["Storage Conditions"].unique()),
            key="plan_storage"
        )

        plan_purchase = st.selectbox(
            "Purchase History",
            sorted(df["Purchase History"].unique()),
            key="plan_purchase"
        )

    with col2:

        plan_season = st.selectbox(
            "Seasonality",
            sorted(df["Seasonality"].unique()),
            key="plan_season"
        )

        plan_preparation = st.selectbox(
            "Preparation Method",
            sorted(df["Preparation Method"].unique()),
            key="plan_preparation"
        )

        plan_location = st.selectbox(
            "Geographical Location",
            sorted(df["Geographical Location"].unique()),
            key="plan_location"
        )

        plan_pricing = st.selectbox(
            "Pricing",
            sorted(df["Pricing"].unique()),
            key="plan_pricing"
        )

    st.divider()

    if st.button(
        "🤖 Generate Smart Food Plan",
        use_container_width=True
    ):

        planning_input = pd.DataFrame([
            {
                "Type of Food": plan_food_type,
                "Number of Guests": plan_guests,
                "Event Type": plan_event,
                "Storage Conditions": plan_storage,
                "Purchase History": plan_purchase,
                "Seasonality": plan_season,
                "Preparation Method": plan_preparation,
                "Geographical Location": plan_location,
                "Pricing": plan_pricing
            }
        ])

        # -----------------------------------------
        # STEP 1 — QUANTITY PREDICTION
        # -----------------------------------------

        predicted_quantity = quantity_model.predict(
            planning_input
        )[0]

        # -----------------------------------------
        # STEP 2 — WASTE PREDICTION
        # -----------------------------------------

        waste_input = planning_input.copy()

        waste_input["Quantity of Food"] = predicted_quantity

        predicted_waste = waste_model.predict(
            waste_input
        )[0]

        # -----------------------------------------
        # WASTE RISK
        # -----------------------------------------

        low_limit = df["Wastage Food Amount"].quantile(0.33)
        high_limit = df["Wastage Food Amount"].quantile(0.67)

        if predicted_waste <= low_limit:
            risk = "LOW"
        elif predicted_waste <= high_limit:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        # -----------------------------------------
        # RESULTS
        # -----------------------------------------

        st.subheader("🤖 ECO Meal AI Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "👥 Expected Guests",
                f"{plan_guests}"
            )

        with r2:
            st.metric(
                "🍽️ AI Estimated Quantity",
                f"{predicted_quantity:.2f}"
            )

        with r3:
            st.metric(
                "♻️ Predicted Food Waste",
                f"{predicted_waste:.2f}"
            )

        st.divider()

        st.subheader("📊 AI Planning Summary")

        summary = pd.DataFrame({
            "Parameter": [
                "Expected Guests",
                "Food Type",
                "Event Type",
                "Preparation Method",
                "AI Estimated Quantity",
                "Predicted Food Waste",
                "Waste Risk"
            ],
            "Value": [
                plan_guests,
                plan_food_type,
                plan_event,
                plan_preparation,
                f"{predicted_quantity:.2f}",
                f"{predicted_waste:.2f}",
                risk
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("💡 Smart Recommendations")

        recommendations = []

        if plan_preparation == "Buffet":
            recommendations.append(
                "Prepare food in controlled batches and refill based "
                "on actual demand."
            )

        elif plan_preparation == "Sit-down Dinner":
            recommendations.append(
                "Use portion planning based on expected guest demand."
            )

        else:
            recommendations.append(
                "Monitor serving demand and avoid unnecessary preparation."
            )

        if plan_storage == "Room Temperature":
            recommendations.append(
                "Minimize holding time and monitor storage conditions."
            )

        if plan_food_type in ["Meat", "Dairy Products"]:
            recommendations.append(
                "Pay particular attention to safe handling and storage "
                "of prepared surplus."
            )

        if risk == "HIGH":
            recommendations.append(
                "The predicted waste is in the higher range of the "
                "historical dataset. Consider reviewing preparation "
                "and serving strategy."
            )

        elif risk == "MEDIUM":
            recommendations.append(
                "The predicted waste is in the medium range of the "
                "historical dataset. Monitor demand and leftovers."
            )

        else:
            recommendations.append(
                "The predicted waste is in the lower range of the "
                "historical dataset."
            )

        for recommendation in recommendations:
            st.success("✓ " + recommendation)

        st.caption(
            "The estimated quantity and waste are model predictions "
            "based on historical data. They are not guaranteed "
            "real-world quantities and the dataset does not establish "
            "a physical unit such as kilograms."
        )


# ============================================================
# TAB 2 — WASTE PREDICTION
# ============================================================

with tab2:

    st.header("🔮 AI Waste Prediction")

    st.write(
        "Enter the expected restaurant or event conditions and "
        "manually specify a preparation quantity."
    )

    col1, col2 = st.columns(2)

    with col1:

        food_type = st.selectbox(
            "Type of Food",
            sorted(df["Type of Food"].unique()),
            key="waste_food_type"
        )

        guests = st.number_input(
            "Number of Guests",
            min_value=1,
            max_value=1000,
            value=350,
            key="waste_guests"
        )

        event_type = st.selectbox(
            "Event Type",
            sorted(df["Event Type"].unique()),
            key="waste_event_type"
        )

        quantity = st.number_input(
            "Quantity of Food",
            min_value=1,
            max_value=1000,
            value=450,
            key="waste_quantity"
        )

        storage = st.selectbox(
            "Storage Conditions",
            sorted(df["Storage Conditions"].unique()),
            key="waste_storage"
        )

    with col2:

        purchase_history = st.selectbox(
            "Purchase History",
            sorted(df["Purchase History"].unique()),
            key="waste_purchase"
        )

        seasonality = st.selectbox(
            "Seasonality",
            sorted(df["Seasonality"].unique()),
            key="waste_season"
        )

        preparation = st.selectbox(
            "Preparation Method",
            sorted(df["Preparation Method"].unique()),
            key="waste_preparation"
        )

        location = st.selectbox(
            "Geographical Location",
            sorted(df["Geographical Location"].unique()),
            key="waste_location"
        )

        pricing = st.selectbox(
            "Pricing",
            sorted(df["Pricing"].unique()),
            key="waste_pricing"
        )

    st.divider()

    if st.button(
        "🔮 Predict Food Waste",
        use_container_width=True
    ):

        input_data = pd.DataFrame([
            {
                "Type of Food": food_type,
                "Number of Guests": guests,
                "Event Type": event_type,
                "Quantity of Food": quantity,
                "Storage Conditions": storage,
                "Purchase History": purchase_history,
                "Seasonality": seasonality,
                "Preparation Method": preparation,
                "Geographical Location": location,
                "Pricing": pricing
            }
        ])

        prediction = waste_model.predict(input_data)[0]

        low_limit = df["Wastage Food Amount"].quantile(0.33)
        high_limit = df["Wastage Food Amount"].quantile(0.67)

        if prediction <= low_limit:
            risk = "LOW"
        elif prediction <= high_limit:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        st.subheader("🤖 AI Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Predicted Waste",
                f"{prediction:.2f}"
            )

        with r2:
            st.metric(
                "Waste Risk",
                risk
            )

        with r3:

            quantity_per_guest = quantity / guests

            st.metric(
                "Food / Guest",
                f"{quantity_per_guest:.2f}"
            )

        st.subheader("💡 Smart Reduction Recommendations")

        recommendations = []

        if quantity_per_guest > 1.2:
            recommendations.append(
                "Reduce preparation quantity relative to expected guests."
            )

        if preparation == "Buffet":
            recommendations.append(
                "Prepare food in smaller batches and monitor buffet demand."
            )

        elif preparation == "Sit-down Dinner":
            recommendations.append(
                "Improve portion planning based on expected guest consumption."
            )

        else:
            recommendations.append(
                "Track serving demand to avoid unnecessary preparation."
            )

        if storage == "Room Temperature":
            recommendations.append(
                "Improve storage monitoring and minimize holding time."
            )

        if food_type in ["Meat", "Dairy Products"]:
            recommendations.append(
                "Monitor prepared surplus carefully and prioritize safe handling."
            )

        recommendations.append(
            "Record actual leftovers after the event to improve future predictions."
        )

        for recommendation in recommendations:
            st.success("✓ " + recommendation)


# ============================================================
# TAB 3 — ANALYTICS
# ============================================================

with tab3:

    st.header("📊 Food Waste Analytics")

    st.write(
        "Insights from the cleaned dataset used to train the prototype."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Records Analyzed",
            f"{len(df):,}"
        )

    with c2:
        st.metric(
            "Average Waste",
            f"{df['Wastage Food Amount'].mean():.2f}"
        )

    with c3:
        st.metric(
            "Minimum Waste",
            f"{df['Wastage Food Amount'].min():.0f}"
        )

    with c4:
        st.metric(
            "Maximum Waste",
            f"{df['Wastage Food Amount'].max():.0f}"
        )

    st.divider()

    st.subheader("🍎 Average Waste by Food Type")

    food_waste = (
        df.groupby("Type of Food")["Wastage Food Amount"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(food_waste)

    st.subheader("🍽️ Average Waste by Preparation Method")

    preparation_waste = (
        df.groupby("Preparation Method")["Wastage Food Amount"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(preparation_waste)

    st.subheader("📈 Food Quantity vs Waste")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(
        df["Quantity of Food"],
        df["Wastage Food Amount"],
        alpha=0.5
    )

    ax.set_xlabel("Quantity of Food")
    ax.set_ylabel("Wastage Food Amount")
    ax.set_title("Quantity of Food vs Waste")

    st.pyplot(fig)

    st.subheader("👥 Guests vs Waste")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(
        df["Number of Guests"],
        df["Wastage Food Amount"],
        alpha=0.5
    )

    ax.set_xlabel("Number of Guests")
    ax.set_ylabel("Wastage Food Amount")
    ax.set_title("Number of Guests vs Waste")

    st.pyplot(fig)


# ============================================================
# TAB 4 — FOOD RECOVERY
# ============================================================

with tab4:

    st.header("♻️ Smart Food Recovery")

    st.write(
        "Identify potential surplus food and guide it through a "
        "food-safety verification and recovery workflow."
    )

    st.info(
        "This module demonstrates the proposed recovery workflow. "
        "The current dataset does not contain real-time food-safety, "
        "expiry, or NGO-partner verification data."
    )

    st.subheader("1️⃣ Surplus Food Identification")

    col1, col2 = st.columns(2)

    with col1:
        surplus = st.number_input(
            "Estimated Surplus Food",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=1.0
        )

    with col2:
        source = st.selectbox(
            "Surplus Source",
            [
                "Prepared Food",
                "Buffet Leftovers",
                "Kitchen Surplus",
                "Event Surplus"
            ]
        )

    st.divider()

    st.subheader("2️⃣ Food-Safety Verification")

    safety_1 = st.checkbox(
        "Food was handled according to applicable food-safety requirements."
    )

    safety_2 = st.checkbox(
        "Food was stored under appropriate conditions."
    )

    safety_3 = st.checkbox(
        "Safe holding time has been verified."
    )

    safety_4 = st.checkbox(
        "Food is suitable for potential redistribution."
    )

    safety_score = sum([
        safety_1,
        safety_2,
        safety_3,
        safety_4
    ])

    st.progress(
        safety_score / 4,
        text=f"Verification Checklist: {safety_score}/4"
    )

    st.divider()

    st.subheader("3️⃣ Recovery Eligibility")

    if surplus <= 0:

        st.warning(
            "No surplus food has been entered."
        )

    elif safety_score == 4:

        st.success(
            "✅ Potential Recovery Candidate"
        )

        st.write(
            f"Estimated surplus: **{surplus:.1f} units**"
        )

        st.write(
            f"Source: **{source}**"
        )

        st.subheader("🤝 Recommended Action")

        st.success(
            "Verify the applicable food-safety requirements and "
            "connect the eligible surplus with an authorized "
            "redistribution partner."
        )

    elif safety_score >= 2:

        st.warning(
            "⚠️ Verification Incomplete"
        )

        st.write(
            "Additional food-safety verification is required "
            "before considering redistribution."
        )

    else:

        st.error(
            "❌ Not Eligible for Recovery"
        )

        st.write(
            "Do not redistribute until the required safety "
            "conditions have been verified."
        )

    st.divider()

    st.subheader("4️⃣ Proposed Recovery Workflow")

    st.markdown(
        """
        **🍽️ Surplus Identified**

        ↓

        **🔍 Food-Safety Verification**

        ↓

        **✅ Eligibility Decision**

        ↓

        **🤝 Authorized Redistribution Partner**

        ↓

        **📦 Safe Distribution**

        **OR**

        **♻️ Responsible Disposal**
        """
    )

    st.caption(
        "Prototype note: The system does not currently verify food "
        "safety, expiry, legal eligibility, or real-time NGO availability."
    )


# ============================================================
# TAB 5 — WASTE REDUCTION SIMULATOR
# ============================================================

with tab5:

    st.header("🎯 Waste Reduction Simulator")

    st.write(
        "Test different preparation quantities and see how the AI "
        "estimates the change in food waste."
    )

    st.subheader("📋 Event Conditions")

    col1, col2 = st.columns(2)

    with col1:

        sim_food_type = st.selectbox(
            "Type of Food",
            sorted(df["Type of Food"].unique()),
            key="sim_food_type"
        )

        sim_guests = st.number_input(
            "Expected Guests",
            min_value=1,
            max_value=1000,
            value=350,
            key="sim_guests"
        )

        sim_event = st.selectbox(
            "Event Type",
            sorted(df["Event Type"].unique()),
            key="sim_event"
        )

        sim_storage = st.selectbox(
            "Storage Conditions",
            sorted(df["Storage Conditions"].unique()),
            key="sim_storage"
        )

        sim_purchase = st.selectbox(
            "Purchase History",
            sorted(df["Purchase History"].unique()),
            key="sim_purchase"
        )

    with col2:

        sim_season = st.selectbox(
            "Seasonality",
            sorted(df["Seasonality"].unique()),
            key="sim_season"
        )

        sim_preparation = st.selectbox(
            "Preparation Method",
            sorted(df["Preparation Method"].unique()),
            key="sim_preparation"
        )

        sim_location = st.selectbox(
            "Geographical Location",
            sorted(df["Geographical Location"].unique()),
            key="sim_location"
        )

        sim_pricing = st.selectbox(
            "Pricing",
            sorted(df["Pricing"].unique()),
            key="sim_pricing"
        )

    st.divider()

    st.subheader("🍽️ Preparation Quantity")

    current_quantity = st.number_input(
        "Current Planned Food Quantity",
        min_value=1.0,
        max_value=1000.0,
        value=450.0,
        step=1.0,
        key="current_quantity"
    )

    reduction_percent = st.slider(
        "Reduce Preparation Quantity (%)",
        min_value=0,
        max_value=30,
        value=10,
        key="reduction_percent"
    )

    simulated_quantity = current_quantity * (
        1 - reduction_percent / 100
    )

    st.info(
        f"AI simulation quantity: **{simulated_quantity:.0f} units** "
        f"after a {reduction_percent}% reduction."
    )

    current_input = pd.DataFrame([{
        "Type of Food": sim_food_type,
        "Number of Guests": sim_guests,
        "Event Type": sim_event,
        "Quantity of Food": current_quantity,
        "Storage Conditions": sim_storage,
        "Purchase History": sim_purchase,
        "Seasonality": sim_season,
        "Preparation Method": sim_preparation,
        "Geographical Location": sim_location,
        "Pricing": sim_pricing
    }])

    simulated_input = current_input.copy()

    simulated_input["Quantity of Food"] = simulated_quantity

    current_waste = waste_model.predict(current_input)[0]

    simulated_waste = waste_model.predict(simulated_input)[0]

    waste_difference = current_waste - simulated_waste

    if current_waste != 0:
        reduction_percentage = (
            waste_difference / current_waste
        ) * 100
    else:
        reduction_percentage = 0

    st.divider()

    st.subheader("🤖 AI Simulation Result")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Current Waste",
            f"{current_waste:.2f}"
        )

    with r2:
        st.metric(
            "Simulated Waste",
            f"{simulated_waste:.2f}"
        )

    with r3:
        st.metric(
            "Estimated Waste Change",
            f"{waste_difference:.2f}"
        )

    with r4:
        st.metric(
            "Estimated Reduction",
            f"{reduction_percentage:.1f}%"
        )

    st.divider()

    st.subheader("📊 Current vs Simulated Plan")

    comparison = pd.DataFrame({
        "Scenario": [
            "Current Plan",
            "AI Simulated Plan"
        ],
        "Food Quantity": [
            current_quantity,
            simulated_quantity
        ],
        "Predicted Waste": [
            current_waste,
            simulated_waste
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("💡 AI Recommendation")

    if waste_difference > 0:

        st.success(
            f"Based on this simulation, reducing preparation quantity "
            f"by {reduction_percent}% is associated with an estimated "
            f"waste reduction of {waste_difference:.2f} units "
            f"under the selected conditions."
        )

    elif waste_difference < 0:

        st.warning(
            "The selected reduction does not produce a lower predicted "
            "waste value in the current model."
        )

    else:

        st.info(
            "The model predicts no change in waste for this simulation."
        )

    st.caption(
        "Note: This is a model-based simulation using the trained "
        "prototype. It is not a guaranteed real-world reduction."
    )


# ============================================================
# TAB 6 — IMPACT DASHBOARD
# ============================================================

with tab6:

    st.header("📈 Impact Dashboard")

    st.write(
        "Dataset-based insights showing the scale and patterns "
        "of food waste in the analyzed records."
    )

    st.subheader("📊 Key Metrics")

    avg_guests = df["Number of Guests"].mean()
    avg_quantity = df["Quantity of Food"].mean()
    avg_waste = df["Wastage Food Amount"].mean()

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Records Analyzed",
            f"{len(df):,}"
        )

    with m2:
        st.metric(
            "Average Guests",
            f"{avg_guests:.0f}"
        )

    with m3:
        st.metric(
            "Average Food Quantity",
            f"{avg_quantity:.1f}"
        )

    with m4:
        st.metric(
            "Average Waste",
            f"{avg_waste:.1f}"
        )

    st.divider()

    st.subheader("📦 Waste Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(
        df["Wastage Food Amount"],
        bins=15,
        edgecolor="black"
    )

    ax.set_xlabel("Wastage Food Amount")
    ax.set_ylabel("Number of Records")
    ax.set_title("Distribution of Food Waste")

    st.pyplot(fig)

    st.subheader("🍎 Average Waste by Food Type")

    food_analysis = (
        df.groupby("Type of Food")["Wastage Food Amount"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )

    food_analysis.columns = [
        "Average Waste",
        "Number of Records"
    ]

    st.dataframe(
        food_analysis.round(2),
        use_container_width=True
    )

    st.subheader("🍽️ Waste by Preparation Method")

    preparation_analysis = (
        df.groupby("Preparation Method")["Wastage Food Amount"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(preparation_analysis)

    st.subheader("🎉 Waste by Event Type")

    event_analysis = (
        df.groupby("Event Type")["Wastage Food Amount"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(event_analysis)

    st.subheader("🔗 Food Quantity vs Waste")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.scatter(
        df["Quantity of Food"],
        df["Wastage Food Amount"],
        alpha=0.5
    )

    ax.set_xlabel("Quantity of Food")
    ax.set_ylabel("Wastage Food Amount")
    ax.set_title("Relationship Between Food Quantity and Waste")

    st.pyplot(fig)

    st.subheader("👥 Guests vs Waste")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.scatter(
        df["Number of Guests"],
        df["Wastage Food Amount"],
        alpha=0.5
    )

    ax.set_xlabel("Number of Guests")
    ax.set_ylabel("Wastage Food Amount")
    ax.set_title("Relationship Between Guests and Waste")

    st.pyplot(fig)

    st.divider()

    st.info(
        "📌 These metrics describe the cleaned dataset used by "
        "the prototype. They should not be interpreted as measured "
        "real-world impact or national food-waste statistics."
    )