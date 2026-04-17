import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt

import streamlit as st
import random
from PIL import Image
import altair as alt
# from htbuilder import HtmlElement, div, hr, a, p, img, styles
# from htbuilder.units import percent, px








data_url = "http://lib.stat.cmu.edu/datasets/boston" 


# data = "C:\Users\DELL\Desktop\streamlit\images\data-processing.png"

# setting up the page streamlit

st.set_page_config(
    page_title="Linear Regression App ", layout="wide", page_icon="./images/linear-regression.png"
)


# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

image_nyu = Image.open('images/nyu.png')
st.image(image_nyu, width=100)

st.title("Linear Regression Lab 🧪")

# navigation dropdown

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction','Visualization','Prediction'])
df = pd.read_csv("marketing_campaign.csv", sep='\t')

list_variables = df.columns
select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)
# page 1 
if app_mode == 'Introduction':
    image_header = Image.open('./images/Linear-Regression1.webp')
    st.image(image_header, width=600)


    st.markdown("### 00 - Show  Dataset")
    st.markdown("### Column Descriptions")
    descriptions = {
        'ID': 'Customer ID',
        'Year_Birth': 'Birth year',
        'Education': 'Education level',
        'Marital_Status': 'Marital status',
        'Income': 'Annual income',
        'Kidhome': 'Number of children',
        'Teenhome': 'Number of teenagers',
        'Dt_Customer': 'Date of enrollment',
        'Recency': 'Days since last purchase',
        'MntWines': 'Amount spent on wine',
        'MntFruits': 'Amount spent on fruits',
        'MntMeatProducts': 'Amount spent on meat',
        'MntFishProducts': 'Amount spent on fish',
        'MntSweetProducts': 'Amount spent on sweets',
        'MntGoldProds': 'Amount spent on gold',
        'NumDealsPurchases': 'Number of purchases with discount',
        'NumWebPurchases': 'Number of web purchases',
        'NumCatalogPurchases': 'Number of catalog purchases',
        'NumStorePurchases': 'Number of store purchases',
        'NumWebVisitsMonth': 'Number of web visits per month',
        'AcceptedCmp3': 'Accepted campaign 3',
        'AcceptedCmp4': 'Accepted campaign 4',
        'AcceptedCmp5': 'Accepted campaign 5',
        'AcceptedCmp1': 'Accepted campaign 1',
        'AcceptedCmp2': 'Accepted campaign 2',
        'Complain': 'Number of complaints',
        'Z_CostContact': 'Cost of contact',
        'Z_Revenue': 'Revenue',
        'Response': 'Response to last campaign'
    }
    for col, desc in descriptions.items():
        st.markdown(f"**{col}**: {desc}") 
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))
    
    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    st.text('(Rows,Columns)')
    st.write(df.shape)


    st.markdown("### 01 - Description")
    st.dataframe(df.describe())



    st.markdown("### 02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df.isnull().sum()
    totalmiss = dfnull.sum().round(2)
    st.write("Number of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 100:
        st.success(f"Looks good! as we have only {totalmiss} missing values.")
    else:
        st.warning("Poor data quality due to large number of missing values.")
        

    # st.markdown("### 03 - Completeness")
    # st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # # st.write("Total data length:", len(df))
    # nonmissing = (df.notnull().sum().round(2))
    # completeness= round(sum(nonmissing)/len(df),2)
    # st.write("Completeness ratio:",completeness)
    # st.write(nonmissing)
    # if completeness >= 0.80:
    #     st.success("Looks good! as we have completeness ratio greater than 0.85.")
           
    # else:
    #     st.success("Poor data quality due to low completeness ratio( less than 0.85).")

    st.markdown("### 03 - Complete Report")
    
    st.subheader("Dataset Overview")
    st.write(df.describe(include='all'))
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    st.subheader("Data Types")
    st.write(df.dtypes)


if app_mode == 'Visualization':
    st.markdown("## Visualization")
    symbols = st.multiselect("Select two variables",list_variables, )
    width1 = st.sidebar.slider("plot width", 1, 25, 10)
    #symbols = st.multiselect("", list_variables, list_variables[:5])
    tab1, tab2= st.tabs(["Line Chart","📈 Correlation"])    

    tab1.subheader("Line Chart")
    st.line_chart(data=df, x=symbols[0],y=symbols[1], use_container_width=True)
    st.write(" ")
    st.bar_chart(data=df, x=symbols[0], y=symbols[1], use_container_width=True)

    tab2.subheader("Correlation Tab 📉")
    numerical_df = df.select_dtypes(include=[np.number])
    fig,ax = plt.subplots(figsize=(width1, width1))
    sns.heatmap(numerical_df.corr(),cmap= sns.cubehelix_palette(8),annot = True, ax=ax)
    tab2.write(fig)


    st.write(" ")
    st.write(" ")
    st.markdown("### Pairplot")

    numerical_vars = df.select_dtypes(include=[np.number]).columns[:5]
    df2 = df[numerical_vars]
    fig3 = sns.pairplot(df2)
    st.pyplot(fig3)




if app_mode == 'Prediction':
    st.markdown("## Prediction")
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    new_df= df.drop(labels=select_variable, axis=1)  #axis=1 means we drop data by columns
    list_var = new_df.columns
    output_multi = st.multiselect("Select Explanatory Variables", list_var)

    def predict(target_choice,train_size,new_df,output_multi):
        #independent variables / explanatory variables
        #choosing column for target
        new_df2 = new_df[output_multi]
        x =  new_df2
        y = df[target_choice]
        col1,col2 = st.columns(2)
        col1.subheader("Feature Columns top 25")
        col1.write(x.head(25))
        col2.subheader("Target Column top 25")
        col2.write(y.head(25))
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)
        lm = LinearRegression()
        lm.fit(X_train,y_train)
        predictions = lm.predict(X_test)

        return X_train, X_test, y_train, y_test, predictions,x,y

    X_train, X_test, y_train, y_test, predictions,x,y= predict(select_variable,train_size,new_df,list_var)

    st.subheader('🎯 Results')


    st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
    st.write("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, predictions ),2))
    st.write("3) MSE: ", np.round(mt.mean_squared_error(y_test, predictions),2))
    st.write("4) The R-Square score of the model is " , np.round(mt.r2_score(y_test, predictions),2))




if __name__=='__main__':
    main()



# def image(src_as_string, **style):
#     return img(src=src_as_string, style=styles(**style))


# def link(link, text, **style):
#     return a(_href=link, _target="_blank", style=styles(**style))(text)


# def layout(*args):

#     style = """
#     <style>
#       # MainMenu {visibility: hidden;}
#       footer {visibility: hidden;background - color: white}
#      .stApp { bottom: 80px; }
#     </style>
#     """
#     style_div = styles(
#         position="fixed",
#         left=0,
#         bottom=0,
#         margin=px(0, 0, 0, 0),
#         width=percent(100),
#         color="black",
#         text_align="center",
#         height="auto",
#         opacity=1,

#     )

#     style_hr = styles(
#         display="block",
#         margin=px(8, 8, "auto", "auto"),
#         border_style="inset",
#         border_width=px(2)
#     )

#     body = p()
#     foot = div(
#         style=style_div
#     )(
#         hr(
#             style=style_hr
#         ),
#         body
#     )

#     st.markdown(style, unsafe_allow_html=True)

#     for arg in args:
#         if isinstance(arg, str):
#             body(arg)

#         elif isinstance(arg, HtmlElement):
#             body(arg)

#     st.markdown(str(foot), unsafe_allow_html=True)

# def footer2():
#     myargs = [
#         "👨🏼‍💻 Made by ",
#         link("https://github.com/NYU-DS-4-Everyone", "NYU - Professor Gaëtan Brison"),
#         "🚀"
#     ]
#     layout(*myargs)


# if __name__ == "__main__":
#     footer2()
