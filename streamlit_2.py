import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Streamlit App
st.sidebar.header('Menu')
menu = st.sidebar.radio('Select an option:', ['Objective and Findings', 'Data Visualization', 'Statistical Tests', 'Predictive Model'])


if menu == 'Objective and Findings':
    st.write('## Objective')
    st.write("""
    The paramount goal of this extensive study is to meticulously scrutinize 
    the variances and drifts inherent in the chronometric measurements 
    throughout diverse production phases, transitioning from Factory A to Factory B. 
    This analysis is pivotal in deciphering the inherent discrepancies and 
    formulating strategies to enhance precision and coherence in the production lifecycle.
    """)
    st.write('## Key Findings')
    st.write("""
    ### Significant Variability:
    The examination unveiled that a substantial 18% of all movements originating from Factory A 
    manifest a variability in measurements, surpassing ±50%, in contrast to those observed in Factory B. 
    This underscores a crucial need to address these inconsistencies to uphold the integrity of the production process.
    """)



elif menu == 'Data Visualization':
    st.write('## Generate Data')
    n = st.number_input('Number of Data Points', min_value=1, value=1000, step=100)
    
    if st.button('Generate DataFrames'):
        np.random.seed(42)
        
        # Create a matched set of observations for Factory A
        measurements = np.random.normal(100, 35, n)
        
        # Adding more variability to the first and last decile of Factory A
        quantiles = np.quantile(measurements, [0.1, 0.9])
        mask = (measurements <= quantiles[0]) | (measurements >= quantiles[1])
        measurements[mask] += np.random.normal(0, 50, np.sum(mask))
        
        factory_a = pd.DataFrame({
            'Measurement': measurements
        })
        factory_b = pd.DataFrame({
            'Measurement': np.random.normal(130, 50, len(factory_a))  # Centered at 130 with higher SD for Factory B
        })

#         factory_b = pd.DataFrame({'Measurement': np.random.normal(100, 50, len(factory_a))})
        
        
        st.session_state.factory_a = factory_a
        st.session_state.factory_b = factory_b
        st.success('DataFrames Generated!')
        
    if 'factory_a' in st.session_state and 'factory_b' in st.session_state:
        fig, ax = plt.subplots()
        ax.hist(st.session_state.factory_a['Measurement'], alpha=0.5, label='Factory A', bins=30, range=[0, 200])
        ax.hist(st.session_state.factory_b['Measurement'], alpha=0.5, label='Factory B', bins=30, range=[0, 200])
        ax.axvline(x=50, color='r', linestyle='dashed', linewidth=2, label='Tolerance Lower Limit')
        ax.axvline(x=150, color='g', linestyle='dashed', linewidth=2, label='Tolerance Upper Limit')
        ax.set_xlabel('Measurement')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')
        st.pyplot(fig)
        
        # Additional Plots for first 10% and last 10% of Factory B and their corresponding values in Factory A.
        quantiles_b = np.quantile(st.session_state.factory_b['Measurement'], [0.1, 0.9])
        
        mask_lower = st.session_state.factory_b['Measurement'] <= quantiles_b[0]
        mask_upper = st.session_state.factory_b['Measurement'] >= quantiles_b[1]
        
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10,8))
        
        axs[0].hist(st.session_state.factory_a[mask_lower]['Measurement'], alpha=0.6, bins=30, range=[0, 200])
        axs[0].set_title('Distribution of the first 10% of Factory B in Factory A')
        axs[0].axvline(x=50, color='r', linestyle='dashed', linewidth=2)
        axs[0].axvline(x=150, color='g', linestyle='dashed', linewidth=2)
        
        axs[1].hist(st.session_state.factory_a[mask_upper]['Measurement'], alpha=0.6, bins=30, range=[0, 200])
        axs[1].set_title('Distribution of the last 10% of Factory B in Factory A')
        axs[1].axvline(x=50, color='r', linestyle='dashed', linewidth=2)
        axs[1].axvline(x=150, color='g', linestyle='dashed', linewidth=2)
        
        st.pyplot(fig)
        
    else:
        st.warning('Please generate DataFrames first!')


#     if 'factory_a' in st.session_state and 'factory_b' in st.session_state:
#         t_stat, p_value = stats.ttest_ind(st.session_state.factory_a['Measurement'], st.session_state.factory_b['Measurement'])
#         st.write('### P-value from t-test:')
#         st.write(p_value)
        
#         if p_value <= 0.05:
#             st.write('### Conclusion:')
#             st.markdown('The p-value is **less than 0.05**, hence we reject the null hypothesis and conclude that the two samples are **statistically different**.')
#         else:
#             st.write('### Conclusion:')
#             st.markdown('The p-value is **greater than 0.05**, hence we do not reject the null hypothesis and conclude that there is **no statistical difference** between the two samples.')
#     else:
#         st.warning('Please generate DataFrames first in the Data Visualization menu!')  


elif menu == 'Predictive Model':
    if 'factory_a' in st.session_state and 'factory_b' in st.session_state:
        # Extracting measurements from Factory A as Features
        X = st.session_state.factory_a[['Measurement']]
        
        # Defining target variable based on Tolerance Limit
        y = (st.session_state.factory_b['Measurement'] < 50) | (st.session_state.factory_b['Measurement'] > 150)
        y = y.astype(int)  # Convert to 0 and 1
        
        # Splitting Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Building Model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions and Performance Metrics
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # probabilities for the positive class
        
        st.write("### Confusion Matrix")
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        st.pyplot(disp.plot().figure_)
        
        st.write("### Additional Metrics")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        st.metric(label="Accuracy", value=f"{accuracy:0.2f}")
        st.metric(label="Precision", value=f"{precision:0.2f}")
        st.metric(label="Recall", value=f"{recall:0.2f}")
        st.metric(label="F1 Score", value=f"{f1:0.2f}")
        
        st.write("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())
else:
    st.warning('Please generate DataFrames first in the Data Visualization menu!')






#     if st.button('Generate DataFrames'):
#         np.random.seed(42)
        
#         measurements = np.random.normal(100, 20, n)
        
#         quantiles = np.quantile(measurements, [0.1, 0.9])
#         mask = (measurements <= quantiles[0]) | (measurements >= quantiles[1])
#         measurements[mask] += np.random.normal(0, 50, np.sum(mask))
        
#         factory_a = pd.DataFrame({'Measurement': measurements})
#         factory_b = pd.DataFrame({'Measurement': np.random.normal(100, 50, len(factory_a))})
        
#         st.session_state.factory_a = factory_a
#         st.session_state.factory_b = factory_b

#         st.success('DataFrames Generated!')

        
#     if 'factory_a' in st.session_state and 'factory_b' in st.session_state:
#         fig, ax = plt.subplots()
#         ax.hist(st.session_state.factory_a['Measurement'], alpha=0.5, label='Factory A', bins=30, range=[0, 200])
#         ax.hist(st.session_state.factory_b['Measurement'], alpha=0.5, label='Factory B', bins=30, range=[0, 200])
#         ax.axvline(x=50, color='r', linestyle='dashed', linewidth=2, label='Tolerance Lower Limit')
#         ax.axvline(x=150, color='g', linestyle='dashed', linewidth=2, label='Tolerance Upper Limit')
#         ax.set_xlabel('Measurement')
#         ax.set_ylabel('Frequency')
#         ax.legend(loc='upper right')
#         st.pyplot(fig)
        
#         # Additional Plots for first 10% and last 10% of Factory B and their corresponding values in Factory A.
#         quantiles_b = np.quantile(st.session_state.factory_b['Measurement'], [0.1, 0.9])
        
#         mask_lower = st.session_state.factory_b['Measurement'] <= quantiles_b[0]
#         mask_upper = st.session_state.factory_b['Measurement'] >= quantiles_b[1]
        
#         fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10,8))
        
#         axs[0].hist(st.session_state.factory_a[mask_lower]['Measurement'], alpha=0.6, bins=30, range=[0, 200])
#         axs[0].set_title('Distribution of the first 10% of Factory B in Factory A')
#         axs[0].axvline(x=50, color='r', linestyle='dashed', linewidth=2)
#         axs[0].axvline(x=150, color='g', linestyle='dashed', linewidth=2)
        
#         axs[1].hist(st.session_state.factory_a[mask_upper]['Measurement'], alpha=0.6, bins=30, range=[0, 200])
#         axs[1].set_title('Distribution of the last 10% of Factory B in Factory A')
#         axs[1].axvline(x=50, color='r', linestyle='dashed', linewidth=2)
#         axs[1].axvline(x=150, color='g', linestyle='dashed', linewidth=2)
        
#         st.pyplot(fig)
        
#     else:
#         st.warning('Please generate DataFrames first!')

#     if 'factory_a' in st.session_state and 'factory_b' in st.session_state:
#         t_stat, p_value = stats.ttest_ind(st.session_state.factory_a['Measurement'], st.session_state.factory_b['Measurement'])
#         st.write('### P-value from t-test:')
#         st.write(p_value)
        
#         if p_value <= 0.05:
#             st.write('### Conclusion:')
#             st.markdown('The p-value is **less than 0.05**, hence we reject the null hypothesis and conclude that the two samples are **statistically different**.')
#         else:
#             st.write('### Conclusion:')
#             st.markdown('The p-value is **greater than 0.05**, hence we do not reject the null hypothesis and conclude that there is **no statistical difference** between the two samples.')
#     else:
#         st.warning('Please generate DataFrames first in the Data Visualization menu!')  


# elif menu == 'Statistical Tests':
   
#     st.title('T-Test Explanation')

#     st.write("""
#     A t-test is a statistical test used to compare the means of two groups to determine if they are significantly different from each other.
#     """)

#     st.header('1. Hypotheses:')
#     st.write("""
#     - Null Hypothesis ($H_0$): The means of the two groups are equal.
#     - Alternative Hypothesis ($H_a$): The means of the two groups are not equal.
#     """)

#     st.header('2. Calculation:')
#     st.latex(r'''
#     t = \frac{{\bar{x}_1 - \bar{x}_2}}{{\sqrt{\left(\frac{s_1^2}{n_1}\right) + \left(\frac{s_2^2}{n_2}\right)}}}
#     ''')
#     st.write("""
#     where $\\bar{x}_1$ and $\\bar{x}_2$ are the sample means, $s_1^2$ and $s_2^2$ are the sample variances, and $n_1$ and $n_2$ are the sample sizes of the two groups.
#     """)

#     st.header('3. Decision Rule:')
#     st.write("""
#     - If the calculated p-value is less than the significance level (commonly 0.05), you reject the null hypothesis.
#     - If the p-value is greater than or equal to the significance level, you fail to reject the null hypothesis.
#     """)

#     st.header('4. Types of T-Test:')
#     st.write("""
#     - One-sample t-test: Compares the mean of one group to a known value.
#     - Two-sample t-test: Compares the means of two independent groups.
#     - Paired t-test: Compares the means of the same group or matched pairs.
#     """)

#     st.header('5. Assumptions:')
#     st.write("""
#     - The data are normally distributed.
#     - The variances of the two populations being compared are equal (for a two-sample t-test).
#     - The observations are independent.
#     """)

#     st.write("""
#     In conclusion, a t-test helps in determining whether there is a significant difference between the means of two groups under the assumption of normality and, in some cases, equal variances.
#     """)

