import streamlit as st
from models import *

st.title('Caretta SVD Recommendation Engine')

st.markdown("""
*Recommend New Products Based on Similar Users to You.*
""")

for i in range(4):
    st.write("")

customerID = st.number_input("Enter Your Customer ID, ie. 17490", step= 1)
desired_n_of_recom = st.number_input("How many recommendations would you like to see? ie. 10", step= 1)

#--------------------------------------------------------------------
if (st.button("Recommend Products")):

    carettaSvd = CarettaSVD(customerID)

    (sales, sales_original, sales_trimmed, trimmed_original_user_products) = carettaSvd.prepare_data_method_one(customerID)

    (ratings_matrix, unique_customers_dict_real_ref, unique_customers_dict_ref_real) = carettaSvd.create_ratings_matrix(
        sales_trimmed)

    (U, S, V) = carettaSvd.get_SVD(ratings_matrix)

    recommendations, already_bought_non_trimmed = carettaSvd.get_recommendations(sales_original, 200, unique_customers_dict_real_ref, unique_customers_dict_ref_real, U, S, V, desired_n_of_recom)

    df = pd.DataFrame({"Recommendations": recommendations})

    st.dataframe(df, width=600, height=900)

    df_2 = pd.DataFrame({"Success": list(set(recommendations) & set(already_bought_non_trimmed))})

    st.dataframe(df_2, width=600, height=900)

    st.markdown(""" *Success Rate:* """)

    st.write(len(list(set(recommendations) & set(already_bought_non_trimmed))) / desired_n_of_recom * 100)

#--------------------------------------------------------------------
