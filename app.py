import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

import joblib

st.title("Wine Quality and Type Dashboard")

wine_colors = {"Red": "#C0392B", "White": "#BE8004"}

# load model
try:
    model = joblib.load("model_folder/best_wine_model.pkl")
    scaler = joblib.load("model_folder/wine_scaler.pkl")
except:
    st.error("Error loading model files")
    st.stop()

# dataset import
white_wine_data = pd.read_csv("wine_data/winequality-white.csv", sep=";")
red_wine_data = pd.read_csv("wine_data/winequality-red.csv", sep=";")


white_wine_data.columns = white_wine_data.columns.str.strip().str.lower().str.replace(" ", "_")
red_wine_data.columns = red_wine_data.columns.str.strip().str.lower().str.replace(" ", "_")

white_wine_data['type'] = 'White'
red_wine_data['type'] = 'Red'

wine_data_frame = pd.concat([white_wine_data, red_wine_data], ignore_index=True)

def group_quality_scores(score):
    if score <= 4:
        return "low"
    elif score <= 6:
        return "medium"
    elif score <= 9:
        return "high"
    else:
        return None

wine_data_frame["quality_group"] = wine_data_frame["quality"].apply(group_quality_scores)


st.sidebar.header("Input Features")

fixed_acidity = st.sidebar.slider("Fixed Acidity", wine_data_frame.iloc[:, 0].min(), wine_data_frame.iloc[:, 0].max(), float(wine_data_frame.iloc[:, 0].median()))
volatile_acidity = st.sidebar.slider("Volatile Acidity",wine_data_frame.iloc[:, 1].min(), wine_data_frame.iloc[:, 1].max(), float(wine_data_frame.iloc[:, 1].median()))
citric_acid = st.sidebar.slider("Citric Acid", wine_data_frame.iloc[:, 2].min(), wine_data_frame.iloc[:, 2].max(), float(wine_data_frame.iloc[:, 2].median()))
residual_sugar = st.sidebar.slider("Residual Sugar", wine_data_frame.iloc[:, 3].min(), wine_data_frame.iloc[:, 3].max(), float(wine_data_frame.iloc[:, 3].median()))
chlorides = st.sidebar.slider("Chlorides",wine_data_frame.iloc[:, 4].min(), wine_data_frame.iloc[:, 4].max(), float(wine_data_frame.iloc[:, 4].median()))
free_sulfur = st.sidebar.slider("Free Sulfur Dioxide", wine_data_frame.iloc[:, 5].min(), wine_data_frame.iloc[:, 5].max(), float(wine_data_frame.iloc[:, 5].median()))
total_sulfur = st.sidebar.slider("Total Sulfur Dioxide", wine_data_frame.iloc[:, 6].min(), wine_data_frame.iloc[:, 6].max(), float(wine_data_frame.iloc[:, 6].median()))
density = st.sidebar.slider("Density", wine_data_frame.iloc[:, 7].min(), wine_data_frame.iloc[:, 7].max(), float(wine_data_frame.iloc[:, 7].median()))
ph = st.sidebar.slider("pH", wine_data_frame.iloc[:, 8].min(), wine_data_frame.iloc[:, 8].max(), float(wine_data_frame.iloc[:, 8].median()))
sulphates = st.sidebar.slider("Sulphates", wine_data_frame.iloc[:, 9].min(), wine_data_frame.iloc[:, 9].max(), float(wine_data_frame.iloc[:, 9].median()))
alcohol = st.sidebar.slider("Alcohol", wine_data_frame.iloc[:, 10].min(), wine_data_frame.iloc[:, 10].max(), float(wine_data_frame.iloc[:, 10].median()))

predict_btn = st.sidebar.button("Predict", use_container_width=True)


input_data = pd.DataFrame([{
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur,
    "total sulfur dioxide": total_sulfur,
    "density": density,
    "pH": ph,
    "sulphates": sulphates,
    "alcohol": alcohol
}])

feature_names = list(wine_data_frame.iloc[:, :11])


st.header("Prediction")

if predict_btn:
    scaled = scaler.transform(input_data.values)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)

    quality_map = {0: "Low", 1: "Medium", 2: "High"}
    color_map = {0: "Red", 1: "White"}

    pred_quality = quality_map[int(pred[0])]
    pred_color = color_map[int(pred[1])]

    st.write("Quality:", pred_quality)
    st.write("Color:", pred_color)

    q_labels = [quality_map[int(c)] for c in model.estimators_[0].classes_]
    c_labels = [color_map[int(c)] for c in model.estimators_[1].classes_]

    fig_pred = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Quality Probabilities", "Color Probabilities"]
    )

    fig_pred.add_trace(go.Bar(
        x=q_labels,
        y=proba[0][0] * 100,
        showlegend=False
    ), row=1, col=1)

    fig_pred.add_trace(go.Bar(
        x=c_labels,
        y=proba[1][0] * 100,
        showlegend=False,
        marker_color=[wine_colors[c] for c in c_labels]
    ), row=1, col=2)

    st.plotly_chart(fig_pred, use_container_width=True)

else:
    st.write("Select values and click predict")


st.header("Dataset Overview")

st.write("Total Samples:", len(wine_data_frame))
st.write("Red Wines:", len(wine_data_frame[wine_data_frame["type"] == "Red"]))
st.write("White Wines:", len(wine_data_frame[wine_data_frame["type"] == "White"]))
st.write("Avg Alcohol:", round(wine_data_frame["alcohol"].mean(), 2))
st.write("Avg Quality Score:", round(wine_data_frame["quality"].mean(), 2))


st.header("Charts")

# quality distribution
q_counts = wine_data_frame.groupby(["type", "quality"]).size().reset_index(name="count")

fig_q = px.bar(
    q_counts,
    x="quality",
    y="count",
    color="type",
    barmode="group",
    color_discrete_map=wine_colors
)

fig_q.update_traces(name="Wine Type", selector={"red":"white"})

st.plotly_chart(fig_q, use_container_width=True)


# pie chart
color_split = wine_data_frame["type"].value_counts().reset_index()
color_split.columns = ["color", "count"]

fig_pie = px.pie(
    color_split,
    names="color",
    values="count",
    color="color",
    color_discrete_map=wine_colors
)

st.plotly_chart(fig_pie, use_container_width=True)


# violin plot
fig_v = px.violin(
    wine_data_frame,
    x="quality_group",
    y="alcohol",
    color="type",
    box=True,
    color_discrete_map=wine_colors
)

st.plotly_chart(fig_v, use_container_width=True)


# heatmap
num_cols = wine_data_frame.select_dtypes(include="number").columns
corr_matrix = wine_data_frame[num_cols].corr()

fig_heat = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale=["#2B0000", "#C0392B", "#BE8004"]
)

st.plotly_chart(fig_heat, use_container_width=True)


st.header("Feature Correlation with Quality")


# stacked charts 
for subset, title, bar_color in [
    (red_wine_data, "Red Wine", "#C0392B"),
    (white_wine_data, "White Wine", "#BE8004"),
]:
    valid = [f for f in feature_names if f in subset.columns] + ["quality"]

    corr_vals = subset[valid].corr()["quality"].drop("quality").sort_values()

    colors = [bar_color if v > 0 else "#999999" for v in corr_vals.values]

    fig_corr = go.Figure(go.Bar(
        x=corr_vals.values,
        y=corr_vals.index,
        orientation="h",
        marker_color=colors
    ))

    fig_corr.add_vline(x=0)
    fig_corr.update_layout(title=title)

    st.plotly_chart(fig_corr, use_container_width=True)

st.write("Wine Quality Predictor")