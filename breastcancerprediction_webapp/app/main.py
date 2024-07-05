import numpy as np
import streamlit as st 
import pickle as p 
import pandas as pd
import plotly.graph_objects as go


def get_clean():
    data=pd.read_csv("C:/Users/GOKUL RAJ/RAJ/projects/streamlit projects/breastcancerprediction_webapp/data/breast_cancer_data.csv")
    data= data.drop(['Unnamed: 32','id'],axis=1)
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    return data

def add_siderbar():
    
    st.sidebar.header("Cell Nuclei Measurements")
    
    data=get_clean()
    
    slider_labels = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ("Radius (SE)", "radius_se"),
    ("Texture (SE)", "texture_se"),
    ("Perimeter (SE)", "perimeter_se"),
    ("Area (SE)", "area_se"),
    ("Smoothness (SE)", "smoothness_se"),
    ("Compactness (SE)", "compactness_se"),
    ("Concavity (SE)", "concavity_se"),
    ("Concave points (SE)", "concave points_se"),
    ("Symmetry (SE)", "symmetry_se"),
    ("Fractal dimension (SE)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]
    

    input_dict = {}
    # label is the column name which we set and key is the original column name and taking the minimum and maximum value from it.
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict


def get_scaled_values(input_dict):
    data=get_clean()
    X=data.drop(['diagnosis'],axis=1)
    scaled_dict={}
    for key,value in input_dict.items():
        max_val=X[key].max()
        min_val=X[key].min()
        scaled_value=(value-min_val)/(max_val-min_val)
        scaled_dict[key]=scaled_value
    return scaled_dict




def get_radar_chart(input_data):

    
    input_data=get_scaled_values(input_data)
    
    categories = ["Radius",
    "Texture",
    "Perimeter",
    "Area",
    "Smoothness",
    "Compactness",
    "Concavity",
    "Concave points",
    "Symmetry",
    "Fractal dimension"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
    r=[
    input_data["radius_mean"],
    input_data["texture_mean"],
    input_data["perimeter_mean"],
    input_data["area_mean"],
    input_data["smoothness_mean"],
    input_data["compactness_mean"],
    input_data["concavity_mean"],
    input_data["concave points_mean"],
    input_data["symmetry_mean"],
    input_data["fractal_dimension_mean"]],
            theta=categories,
            fill='toself',
            fillcolor='rgba(255, 215, 0, 0.7)',
            name='Mean Value'
        ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data["radius_worst"],
            input_data["texture_worst"],
            input_data["perimeter_worst"],
            input_data["area_worst"],
            input_data["smoothness_worst"],
            input_data["compactness_worst"],
            input_data["concavity_worst"],
            input_data["concave points_worst"],
            input_data["symmetry_worst"],
            input_data["fractal_dimension_worst"]
            ],
        theta=categories,
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.7)' ,
        name='Worst'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r= [ input_data["radius_se"],
    input_data["texture_se"],
    input_data["perimeter_se"],
    input_data["area_se"],
    input_data["smoothness_se"],
    input_data["compactness_se"],
    input_data["concavity_se"],
    input_data["concave points_se"],
    input_data["symmetry_se"],
    input_data["fractal_dimension_se"]],
        theta=categories,
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.7)',
        name='Standard Error'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
                )),
        showlegend=True
        )
    return fig  
    
def add_predictions(input_data):
    model= p.load(open("C:/Users/GOKUL RAJ/RAJ/projects/streamlit projects/breastcancerprediction_webapp/model/model.pkl","rb"))
    scaler= p.load(open("C:/Users/GOKUL RAJ/RAJ/projects/streamlit projects/breastcancerprediction_webapp/model/scaler.pkl","rb"))
    
    input_array=np.array(list(input_data.values())).reshape(1,-1)
    
    input_array_scaled=scaler.transform(input_array)
    prediction=model.predict(input_array_scaled)
    benign_prob = model.predict_proba(input_array_scaled)[0][0]
    malicious_prob = model.predict_proba(input_array_scaled)[0][1]
    if prediction[0]==0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
        #st.write("Benign")
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
        #st.write("Malicious")
        
    st.write("Probability of being benign: ",benign_prob)
    st.write("Probability of being malicious: ",malicious_prob)
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")




def main():
    st.set_page_config(
        page_title="BREAST CANCER PREDICTION",
        page_icon="👩‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={'Get Help': 'https://www.linkedin.com/in/gokulraj075/',
                    'Report a bug': "https://www.linkedin.com/in/gokulraj075/",
                    'About': " GOKUL RAJ.S |Junior Data Scientist | Proficient in Power BI, Advanced SQL, Python | Excel Expert | AZ-104 Certified | AZ-900 Certified | Google Cloud Certified 17+ Badges | Machine Learning | VIT 25"      
                    }
    )
    
    with open("C:/Users/GOKUL RAJ/RAJ/projects/streamlit projects/breastcancerprediction_webapp/assests/style.css") as f:
       st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)
    
    input_data=add_siderbar()
    #st.write(input_data)
    
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Connect this application to your cytology lab to assist in diagnosing breast cancer from tissue samples. Using a machine learning model, the app predicts whether a breast mass is benign or malignant based on measurements received from the cytology lab. You can also manually adjust the measurements using the sliders in the sidebar.")
    col1, col2 = st.columns([4,1])        
    
    with col1:
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        st.subheader("Cell cluster prediction") 
        st.write("The cell cluster is: ") 
        add_predictions(input_data)


if __name__=='__main__':
    main()