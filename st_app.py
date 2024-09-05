import joblib 
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

model = joblib.load('model.pkl')
df = pd.read_csv('data/Salary_Data.csv')
df_pred = pd.read_csv('y_pred.csv')
def about_project():
    st.title("Salary of Jobs")
    st.header('this website can be predict the salary of many jobs')
    st.image('https://th.bing.com/th/id/R.37d9c5e34d119e11770f4c744f47658c?rik=0dNF8W2%2bzwFsXg&riu=http%3a%2f%2fwww1.cv-ag.com%2fwp-content%2fuploads%2f2016%2f01%2fbusiness-people-working-together-istock_000017346252medium.jpg&ehk=VgK9hHrREyweK%2buDFv5XA0PyJZy3QvmVKowBA0p2Zlg%3d&risl=&pid=ImgRaw&r=0')
    






def predict(Age,Gender , Education_Level, Job_Title , Years_of_Experience):
    pred = model.predict([[Age,Gender , Education_Level, Job_Title , Years_of_Experience]])
    plots(round(pred[0], 2))
    

def inputs():
    st.title("Predict Salary")
    Age = st.number_input('Age')
    Gender = st.selectbox('Gender' , ['Male','Female'])
    Education_Level = st.selectbox('Education Level' , ["Bachelor's Degree","High School" ,"Master's Degree","PhD"])
    Job_Title = st.selectbox('Job Title' , ['Software Engineer Manager','Full Stack Engineer','Senior Project Engineer','Senior Software Engineer',
                                           'Data Scientist','Back end Developer','Software Engineer','Front end Developer','Marketing Manager',
                                           'Product Manager','Data Analyst','Web Developer','Financial Manager','Director of HR',
                                            'Director of Marketing','Marketing Coordinator','Junior Sales Associate','Content Marketing Manager',
                                           'Software Developer','Operations Manager','Junior HR Generalist','Sales Manager','Sales Representative',
                                           'Senior Product Marketing Manager','Sales Associate','Senior HR Generalist','Junior Sales Representative',
                                           'Sales Executive','Digital Marketing Manager','Junior Web Developer','Junior HR Coordinator',
                                           'Senior Human Resources Manager','Sales Director','Marketing Analyst','Research Scientist',
                                           'Human Resources Manager','Financial Analyst','Human Resources Coordinator','Research Director',
                                           'Product Designer','Director of Operations','Senior Marketing Manager','Project Manager','Junior Software Developer',
                                           'Junior Software Engineer','Junior Marketing Manager','Graphic Designer','Senior Project Manager',
                                           'Director of Data Science','Senior Financial Analyst','Senior Business Analyst','Marketing Director',
                                           'Junior Financial Analyst','Junior Marketing Coordinator','Senior Product Manager','Senior Research Scientist',
                                           'Senior Operations Manager','Junior Business Development Associate','Senior Business Development Manager'])
    Years_of_Experience = st.number_input('Years of Experience')
    st.button("Predict", on_click=predict, args=(Age,Gender , Education_Level, Job_Title , Years_of_Experience))

def plots(result=0 , ):
    # Plotting
    st.title(f"Salary of Job")
    
    st.markdown(f"salary is : **{result}**")

    st.button("Make Another Prediction", on_click=inputs)
    fig = plt.figure(figsize=(4,4))
    plt.scatter(data= df_pred, x = 'y_test',y = 'y_pred')
    plt.plot([df_pred['y_test'].min(), df_pred['y_test'].max()], [df_pred['y_test'].min(), df_pred ['y_test'].max()])
    plt.title(f'True vs. Predicted Salary (model Regressor)')
    st.pyplot(fig)

page = st.sidebar.selectbox("Select page", ["Predict",'plots','About'])
if page == 'Predict' :
    inputs()
elif page == 'plots':
    plots()
elif page == 'About' :
    about_project()
    