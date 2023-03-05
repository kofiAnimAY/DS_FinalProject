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
import plotly.express as px
import streamlit as st
import random
from PIL import Image
import altair as alt





data_url = "http://lib.stat.cmu.edu/datasets/boston" 


# data = "C:\Users\DELL\Desktop\streamlit\images\data-processing.png"

# setting up the page streamlit
ilr = "./images/linear-regression.png"
st.set_page_config(
    page_title="Linear Regression App ", layout="wide", page_icon=ilr
)

st.title("Understanding Linear Regression💡")

# navigation dropdown
app_mode = st.sidebar.selectbox('Select Page',['Background','Main App'])
# page 1 
if app_mode == 'Background':
    image = Image.open('images\Linear-Regression1.webp')

    st.image(image, caption='Into the ML World:')
    st.subheader('🎯 Definition') 
    st.write(" Linear regression is a statistical technique used to model the relationship between a dependent variable"
              " and one or more independent variables. The technique assumes that the relationship between the variables is linear, meaning that" 
              " a change in one variable is proportional to a change in the other variable(s)")
    st.subheader(" ")
    st.subheader('🎯 Objectives of Linear Regression:') 
    st.write('* Estimate the relationshop between the output variable (or response) and input (or predictor/explanatory) variable')
    st.write('* To predict the value of an output variable based on the value of an input variable')

    st.write("* also to help you understand better --------- for students as joke intended? write or not write")
    st.subheader(" ")
    st.subheader('🎯 Use Cases:')
    st.write("* Finance")
    st.write("* Marketing")
    st.write("* Economics")
    st.write("* Social Science")
    st.write("* Engineering")
    st.subheader(" ")
    st.subheader(' Let\'s get started 🚀') 

#page 2
if app_mode == 'Main App':
    st.sidebar.title('🗃️ Choosing Data Source')
    st.sidebar.markdown('Let\'s start!!')
    with st.sidebar:
        with st.expander(" Select Dataset 💾"):
            choose_data = st.radio(
                "Choose Toy dataset or real world dataset 💾",
                ('Toy dataset - Diabetes', 'real world'))
            
            if choose_data == 'Toy dataset - Diabetes':
                add_dataset = "Diabetes"
            else:
                add_dataset = st.selectbox(
                    "Which dataset would you like to be select?",
                    ("Wine Quality","California Housing")
                )
        with st.expander("Do you have your own dataset? Upload your data here!! 👇"):
            upload_dataset = st.file_uploader("Choose a file")


    #loading the dataset with cache
    @st.cache_data
    def load(data):   
        if add_dataset == "Diabetes":
            data = load_diabetes()
            db_df = pd.DataFrame(data.data, columns=data.feature_names)
            db_df['target'] = data.target

        elif add_dataset == "California Housing":
            data = fetch_california_housing()
            db_df = pd.DataFrame(data.data, columns=data.feature_names)
            # adding the target column
            db_df['target'] = data.target

        # elif add_dataset == "Boston":
        #     db_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        
        # elif add_dataset == "Medical Cost: Insurance":
        #     #data = fetch_california_housing()
        #     db_df = pd.read_csv('dataset\insurance.csv')
        #     cat_cols = ["sex","children","smoker","region"]
        #     df_encode = pd.get_dummies(data = db_df, prefix ='OHE', prefix_sep= '_',
        #                             columns= cat_cols,drop_first= True, dtype= 'int8')
        elif add_dataset == "Wine Quality":
            db_df = pd.read_csv('dataset\winequality-red.csv')

        return db_df
    db_df = load(data = add_dataset)

    #displaying dataframe
    def displaying_df(db_df):
        if st.sidebar.checkbox("Display data 💾", False):
            st.subheader('Checking out the datset 👀')
            if upload_dataset is not None:
                # checking excel or csv
                if 'csv' in upload_dataset.name:
                    db_df = pd.read_csv(upload_dataset)
                else:
                    db_df = pd.read_excel(upload_dataset)
                # checking if any column is not integer or float. That means not clean
                c=0
                c = sum(1 for i in db_df.dtypes if (i!= int and i!= float))
                if c>1:
                    st.error("Unclean dataset will produce error. Clean dataset first!")
                #length of the db
                length = st.slider("Change the size", 0, len(db_df),7)
                st.write(upload_dataset)

                #showing upload datset name upload_dataset.name.split(".")[0] 
                st.subheader("Showing " + upload_dataset.name.split(".")[0] + f" top {length} Dataset 💾")
                st.dataframe(db_df.head(length), 1300, 300)
                

            # choosing sklearn or local dataset
            else:
                #length of the db
                length = st.slider("Change the size", 0, len(db_df),25)

                st.subheader("Showing " + add_dataset + f" top {length} Dataset 💾")
                st.dataframe(db_df.head(length), 1300, 500)
                with st.expander("🔤 Description"):
                    if add_dataset == "Diabetes":
                        col1, col2, col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)
                        col1.markdown(" **Age** ")
                        col1.markdown("age in years")
                        col2.markdown(" **bmi** ")
                        col2.markdown("body mass index")
                        col3.markdown(" **bp** ")
                        col3.markdown("average blood pressure")
                        col4.markdown(" **s1** ")
                        col4.markdown("tc: total serum cholesterol")
                        col5.markdown(" **s2** ")
                        col5.markdown("ldl:low-density lipoproteins")
                        col6.markdown(" **s3** ")
                        col6.markdown("hdl:high-density lipoproteins")
                        col7.markdown(" **s4** ")
                        col7.markdown("tch:, total cholesterol / HDL")
                        col8.markdown(" **s5** ")
                        col8.markdown("ltg, possibly log of serum triglycerides level")
                        col9.markdown(" **s6** ")
                        col9.markdown("glu, blood sugar level")
                        col10.markdown("target")
                        col10.markdown("Disease Progression")

                    if add_dataset == "Wine Quality":
                        col1, col2, col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)
                        col1.markdown(" **fixed acidity** ")
                        col1.markdown("most acids involved with wine or fixed or nonvolatile (do not evaporate readily)")
                        col2.markdown(" **volatile acidity** ")
                        col2.markdown("the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste")
                        col3.markdown(" **citric acid** ")
                        col3.markdown("found in small quantities, citric acid can add 'freshness' and flavor to wines")
                        col4.markdown(" **residual sugar** ")
                        col4.markdown("the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter")
                        col5.markdown(" **chlorides** ")
                        col5.markdown("the amount of salt in the wine")
                        col6.markdown(" **free sulfur dioxide** ")
                        col6.markdown("the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents ")
                        col7.markdown(" **total sulfur dioxide** ")
                        col7.markdown("amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 ")
                        col8.markdown(" **density** ")
                        col8.markdown("the density of water is close to that of water depending on the percent alcohol and sugar content")
                        col9.markdown(" **pH** ")
                        col9.markdown("describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the ")
                        col10.markdown(" **sulphates** ")
                        col10.markdown("a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobia")
                    
                    if add_dataset == "California Housing":
                        col1, col2, col3,col4,col5,col6,col7,col8 = st.columns(8)
                        col1.markdown(" **MedInc** ")
                        col1.markdown("median income in block group")
                        col2.markdown(" **HouseAge** ")
                        col2.markdown("median house age in block group")
                        col3.markdown(" **AveRooms** ")
                        col3.markdown("average number of rooms per household")
                        col4.markdown(" **AveBedrms ** ")
                        col4.markdown("average number of bedrooms per household")
                        col5.markdown(" **Population** ")
                        col5.markdown("block group population")
                        col6.markdown(" **AveOccup ** ")
                        col6.markdown("average number of household members")
                        col7.markdown(" **Latitude** ")
                        col7.markdown("block group latitude")
                        col8.markdown(" **Longitude** ")
                        col8.markdown("block group longitude")
        return db_df
    db_df = displaying_df(db_df)

    # display describe
    if st.sidebar.checkbox("Display describe 🔤", False):
        width0 = st.sidebar.slider("plot width", 200, 1000, 700)
        height0 = st.sidebar.slider("plot height", 200, 1000, 500)
        tab1, tab2 = st.tabs(["🗃 Data","📈 Chart"])   
        tab1.subheader(" Describe 🔤")
        tab1.write(db_df.describe())

        diff_plot = tab2.selectbox(
            "Which plot would you like to be Select?",
            ( "Bar plot", "scatter plot","pair plot")
        )
        if diff_plot == "Bar plot":
            value = tab2.selectbox('Choose value',db_df.columns)
            # chart = alt.Chart(db_df).mark_bar().encode(
            #     x=value,
            #     y="count()",
            #     tooltip=[value]
            # ).interactive()
            # tab2.altair_chart(chart, theme="streamlit", use_container_width=True)

            plot1 = px.histogram(data_frame=db_df, x=value)
            plot1.update_layout(
                title='Count Chart',
                xaxis_title=value,
                yaxis_title='Count'
            )
            plot1.update_layout(height=height0,width= width0)
            tab2.plotly_chart(plot1)
            
        if diff_plot == "scatter plot":
            x_axis = tab2.selectbox('X axis', options=db_df.columns)
            y_axis = tab2.selectbox('Y axis', options=db_df.columns)
            plot = px.scatter(data_frame=db_df, x=x_axis, y=y_axis)
            plot.update_layout(
                title='Scatter Chart',
                xaxis_title=x_axis,
                yaxis_title=y_axis
            )
            plot.update_layout(height=height0,width= width0)
            tab2.plotly_chart(plot)

        if diff_plot == "pair plot":          
            plot4 = px.scatter_matrix(db_df)
            tab2.pyplot(plot4)

    #displaying correlation
    if st.sidebar.checkbox("Display Correlation ✨ ", False):
        tab1, tab2 = st.tabs(["🗃 Data","📈 Chart"])    
        tab1.subheader("Data Tab 💾")
        corr = db_df.corr()
        tab1.dataframe(corr, 1000)

        tab2.subheader("Chart Tab 📉")
        width1 = st.sidebar.slider("plot width", 1, 25, 10)
        height1 = st.sidebar.slider("plot height", 1, 25, 5)
        fig,ax = plt.subplots(figsize=(width1, height1))
        sns.heatmap(db_df.corr(),cmap= sns.cubehelix_palette(8),annot = True, ax=ax)
        tab2.write(fig)


    # choosing the target and the feature column 
    target_choice = st.selectbox('Select your target', db_df.columns)

    # preprocess 
    #def preprocess

    #prediction
    def predict(target_choice):
        #independent variables / explanatory variables
        #choosing column for target
        x = db_df.drop(labels=target_choice, axis=1)  #axis=1 means we drop data by columns
        y = db_df[target_choice]
        col1,col2 = st.columns(2)
        col1.subheader("Feature Columns top 25")
        col1.write(x.head(25))
        col2.subheader("Target Column top 25")
        col2.write(y.head(25))

        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
        lm = LinearRegression()
        lm.fit(X_train,y_train)
        predictions = lm.predict(X_test)

        return X_train, X_test, y_train, y_test, predictions,x,y
    X_train, X_test, y_train, y_test, predictions,x,y= predict(target_choice)

    st.subheader('🎯 Result')

    # calculate these metrics easy-peasy
    tab3, tab4 = st.tabs(["🗃 Data","📈 Chart"]) 
    with tab3.expander("🔤 Description"):

        st.write("* Mean Absolute Error (MAE): Measures the average absolute error.")
        st.write("* Mean Squared Error (MSE): Measures the average squared difference between forecasts and true values.")
        st.write("* Root Mean Squared Error (RMSE): Square root of the MSE. This metric is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.")
    
    # callback to update emojis in Session State
    # in response to the on_click event
    def random_emoji():
        if "emoji" not in st.session_state:
            st.session_state.emoji = random.choice(emojis)
        
    if "emoji" not in st.session_state:
        st.session_state.emoji = "👈"
    else:
        st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target w.r.t features is")
        st.write("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, predictions ),2))
        st.write("3) MSE: ", np.round(mt.mean_squared_error(y_test, predictions),2))
        st.write("4) The R-Square score of the model is " , np.round(mt.r2_score(y_test, predictions),2))



    emojis = ["🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼"]

    show_metrics = st.button(f"Show metrics {st.session_state.emoji}", on_click=random_emoji)



    # choose plot
    width_pred = tab4.slider("plot width", 300, 1000, 700)
    height_pred = tab4.slider("plot height", 300, 1000, 500)

    @st.cache_data 
    def make_plot(width, height,y_test,predictions ):
        # new df for pred and y_test
        # data_join = pd.DataFrame({'y_test': y_test, 'predictions': predictions})
        # chart1 = alt.Chart(data_join).mark_circle().encode(
        # x='y_test',
        # y='predictions',
        # )
        
        # # Set the chart title and axis labels
        # chart1 = chart1.properties(
        #     title='Test vs Prediction',
        #     width=width,
        #     height=height
        # ).interactive()
        # tab2.altair_chart(chart1, theme="streamlit", use_container_width=True)
        data_join = pd.DataFrame({'y_test': y_test, 'predictions': predictions})
        plot2 = px.scatter(data_frame=data_join, x=y_test, y=predictions)
        plot2.update_layout(
            title='Scatter Chart',
            xaxis_title="Y Test",
            yaxis_title="Predictions"
        )
        plot2.update_layout(height=height,width= width)
        tab4.plotly_chart(plot2)
    
        return data_join
    
    data_join = make_plot(width_pred, height_pred,y_test,predictions)







    #coefficients

    # if __name__ == '__main__':
    #     main()