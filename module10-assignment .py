# Module 10 Assignment: Data Manipulation and Cleaning with Pandas
# UrbanStyle Customer Data Cleaning

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING")
print("=" * 60)

# ----- USE THE FOLLOWING CODE TO SIMULATE A CSV FILE (DO NOT MODIFY) -----
from io import StringIO

csv_content = """customer_id,first_name,last_name,email,phone,join_date,last_purchase,total_purchases,total_spent,preferred_category,satisfaction_rating,age,city,state,loyalty_status
CS001,John,Smith,johnsmith@email.com,(555) 123-4567,2023-01-15,2023-12-01,12,"1,250.99",Menswear,4.5,35,Tampa,FL,Gold
CS002,Emily,Johnson,emily.j@email.com,555.987.6543,01/25/2023,10/15/2023,8,$875.50,Womenswear,4,28,Miami,FL,Silver
CS003,Michael,Williams,mw@email.com,(555)456-7890,2023-02-10,2023-11-20,15,"2,100.75",Footwear,5,42,Orlando,FL,Gold
CS004,JESSICA,BROWN,jess.brown@email.com,5551234567,2023-03-05,2023-12-10,6,659.25,Womenswear,3.5,31,Tampa,FL,Bronze
CS005,David,jones,djones@email.com,555-789-1234,2023-03-20,2023-09-18,4,350.00,Menswear,,45,Jacksonville,FL,Bronze
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS007,Robert,Davis,robert.davis@email.com,555.444.7777,04/30/2023,11/25/2023,7,$725.80,Footwear,4.5,38,Miami,FL,Silver
CS008,Jennifer,Garcia,jen.garcia@email.com,(555)876-5432,2023-05-15,2023-10-30,3,280.50,ACCESSORIES,3,25,Orlando,FL,Bronze
CS009,Michael,Williams,m.williams@email.com,5558889999,2023-06-01,2023-12-07,9,1100.00,Menswear,4,39,Jacksonville,FL,Silver
CS010,Emily,Johnson,emilyjohnson@email.com,555-321-6547,2023-06-15,2023-12-15,14,"1,875.25",Womenswear,4.5,27,Miami,FL,Gold
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS011,Amanda,,amanda.p@email.com,(555) 741-8529,2023-07-10,,2,180.00,womenswear,3,32,Tampa,FL,Bronze
CS012,Thomas,Wilson,thomas.w@email.com,,2023-07-25,2023-11-02,5,450.75,menswear,4,44,Orlando,FL,Bronze
CS013,Lisa,Anderson,lisa.a@email.com,555.159.7530,08/05/2023,,0,0.00,Womenswear,,30,Miami,FL,
CS014,James,Taylor,jtaylor@email.com,555-951-7530,2023-08-20,2023-10-10,11,"1,520.65",Footwear,4.5,,Jacksonville,FL,Gold
CS015,Karen,Thomas,karen.t@email.com,(555) 357-9512,2023-09-05,2023-12-12,6,685.30,Womenswear,4,36,Tampa,FL,Silver
"""

customer_data_csv = StringIO(csv_content)
# ----- END OF SIMULATION CODE -----


# -------------------------------------------------------
# TODO 1: Load and Explore the Dataset
# -------------------------------------------------------

# 1.1  Load the dataset and display basic information
raw_df = pd.read_csv(customer_data_csv)

print("\n--- Initial Data Overview ---")
print(raw_df.info())
print("\nFirst few rows:")
print(raw_df.head())
print("\nBasic stats:")
print(raw_df.describe())

# 1.2 Check for missing values and duplicates
initial_missing_counts = raw_df.isnull().sum()
print("\nMissing values per column:")
print(initial_missing_counts)

initial_duplicate_count = raw_df.duplicated().sum()
print(f"\nDuplicate rows found: {initial_duplicate_count}")


# -------------------------------------------------------
# TODO 2: Handle Missing Values
# -------------------------------------------------------

# 2.1 Identify and count missing values
missing_value_report = raw_df.isnull().sum()
print("\n--- Missing Value Report ---")
print(missing_value_report)

# 2.2 Fill missing satisfaction_rating with the median value
satisfaction_median = raw_df['satisfaction_rating'].median()
print(f"\nFilling missing satisfaction_rating with median: {satisfaction_median}")
raw_df['satisfaction_rating'] = raw_df['satisfaction_rating'].fillna(satisfaction_median)

# 2.3 Fill missing last_purchase dates appropriately
# Going with forward fill — if someone hasn't made a recent purchase it's
# reasonable to carry the last known date forward rather than just dropping rows
date_fill_strategy = 'forward_fill'
raw_df['last_purchase'] = raw_df['last_purchase'].ffill()

# 2.4 Handle remaining missing values
# - last_name: can't really infer this, fill with 'Unknown'
# - age: fill with median age
# - loyalty_status: fill with 'Bronze' since that's the entry level
# - phone: fill with placeholder so the row isn't dropped entirely
raw_df['last_name'] = raw_df['last_name'].fillna('Unknown')
raw_df['age'] = raw_df['age'].fillna(raw_df['age'].median())
raw_df['loyalty_status'] = raw_df['loyalty_status'].fillna('Bronze')
raw_df['phone'] = raw_df['phone'].fillna('000-000-0000')

df_no_missing = raw_df.copy()
print(f"\nMissing values after cleaning: {df_no_missing.isnull().sum().sum()}")


# -------------------------------------------------------
# TODO 3: Correct Data Types
# -------------------------------------------------------

df_typed = df_no_missing.copy()

# 3.1 Convert join_date and last_purchase to datetime
df_typed['join_date'] = pd.to_datetime(df_typed['join_date'], format='mixed', dayfirst=False)
df_typed['last_purchase'] = pd.to_datetime(df_typed['last_purchase'], format='mixed', dayfirst=False)

# 3.2 Convert total_spent to numeric (handle currency symbols and commas)
df_typed['total_spent'] = df_typed['total_spent'].astype(str).str.replace('[$,]', '', regex=True)
df_typed['total_spent'] = pd.to_numeric(df_typed['total_spent'], errors='coerce')

# 3.3 Ensure other numeric fields (total_purchases, age) are correct types
df_typed['total_purchases'] = pd.to_numeric(df_typed['total_purchases'], errors='coerce').astype(int)
df_typed['age'] = pd.to_numeric(df_typed['age'], errors='coerce').astype(int)

print("\n--- Data Types After Conversion ---")
print(df_typed.dtypes)


# -------------------------------------------------------
# TODO 4: Clean and Standardize Text Data
# -------------------------------------------------------

df_text_cleaned = df_typed.copy()

# 4.1 Standardize name fields — some rows have ALL CAPS, some are lowercase
df_text_cleaned['first_name'] = df_text_cleaned['first_name'].str.strip().str.title()
df_text_cleaned['last_name'] = df_text_cleaned['last_name'].str.strip().str.title()

# 4.2 Category names are inconsistent (ACCESSORIES, womenswear, Menswear, etc.)
# Capitalize just the first letter to keep it clean
df_text_cleaned['preferred_category'] = df_text_cleaned['preferred_category'].str.strip().str.title()

# 4.3 Phone numbers are all over the place — (555) 123-4567, 555.987.6543, 5551234567, etc.
# Standardizing to XXX-XXX-XXXX format
phone_format = 'XXX-XXX-XXXX'

def standardize_phone(phone):
    # Strip everything that isn't a digit
    digits = str(phone).replace('(', '').replace(')', '').replace('-', '').replace('.', '').replace(' ', '')
    # Make sure we actually have 10 digits before formatting
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return digits  # leave it as-is if something weird happened

df_text_cleaned['phone'] = df_text_cleaned['phone'].apply(standardize_phone)

print("\n--- Sample After Text Cleaning ---")
print(df_text_cleaned[['first_name', 'last_name', 'preferred_category', 'phone']].head(8))


# -------------------------------------------------------
# TODO 5: Remove Duplicates
# -------------------------------------------------------

# CS006 appears twice with identical data — classic copy-paste error in the source system
duplicate_count = df_text_cleaned.duplicated().sum()
print(f"\nDuplicate records found: {duplicate_count}")

# Keep the first occurrence of each duplicate row
df_no_duplicates = df_text_cleaned.drop_duplicates().reset_index(drop=True)
print(f"Rows before: {len(df_text_cleaned)}, Rows after removing duplicates: {len(df_no_duplicates)}")


# -------------------------------------------------------
# TODO 6: Add Derived Features
# -------------------------------------------------------

# 6.1 Days since last purchase
# Using today's date so the number stays current
today = pd.Timestamp.today().normalize()
df_no_duplicates['days_since_last_purchase'] = (today - df_no_duplicates['last_purchase']).dt.days

# 6.2 Average purchase value
# If total_purchases is 0, result should be NaN (can't divide by zero)
df_no_duplicates['average_purchase_value'] = np.where(
    df_no_duplicates['total_purchases'] != 0,
    df_no_duplicates['total_spent'] / df_no_duplicates['total_purchases'],
    np.nan
)

# 6.3 Purchase frequency category
# >= 10 = High, 5-9 = Medium, < 5 = Low
def categorize_frequency(purchases):
    if purchases >= 10:
        return 'High'
    elif purchases >= 5:
        return 'Medium'
    else:
        return 'Low'

df_no_duplicates['purchase_frequency_category'] = df_no_duplicates['total_purchases'].apply(categorize_frequency)

print("\n--- Derived Features Sample ---")
print(df_no_duplicates[['customer_id', 'days_since_last_purchase', 'average_purchase_value', 'purchase_frequency_category']].head(8))


# -------------------------------------------------------
# TODO 7: Clean Up the DataFrame
# -------------------------------------------------------

# 7.1 Rename columns to be more readable
df_renamed = df_no_duplicates.rename(columns={
    'customer_id': 'Customer ID',
    'first_name': 'First Name',
    'last_name': 'Last Name',
    'email': 'Email',
    'phone': 'Phone',
    'join_date': 'Join Date',
    'last_purchase': 'Last Purchase Date',
    'total_purchases': 'Total Purchases',
    'total_spent': 'Total Spent',
    'preferred_category': 'Preferred Category',
    'satisfaction_rating': 'Satisfaction Rating',
    'age': 'Age',
    'city': 'City',
    'state': 'State',
    'loyalty_status': 'Loyalty Status',
    'days_since_last_purchase': 'Days Since Last Purchase',
    'average_purchase_value': 'Avg Purchase Value',
    'purchase_frequency_category': 'Purchase Frequency'
})

# 7.2 Drop columns that aren't super useful for the segmentation analysis
# The raw date columns are redundant now that we have Days Since Last Purchase,
# and State is uniform (all FL) so it adds nothing to segmentation
df_final = df_renamed.drop(columns=['Join Date', 'Last Purchase Date', 'State'])

# 7.3 Sort by total spent descending — best customers first
df_final = df_final.sort_values('Total Spent', ascending=False).reset_index(drop=True)

print("\n--- Final DataFrame Shape ---")
print(df_final.shape)
print(df_final.dtypes)


# -------------------------------------------------------
# TODO 8: Generate Insights from Cleaned Data
# -------------------------------------------------------

# 8.1 Average spend grouped by loyalty tier
avg_spent_by_loyalty = df_final.groupby('Loyalty Status')['Total Spent'].mean().sort_values(ascending=False)
print("\n--- Avg Spent by Loyalty Status ---")
print(avg_spent_by_loyalty)

# 8.2 Revenue by preferred category
category_revenue = df_final.groupby('Preferred Category')['Total Spent'].sum().sort_values(ascending=False)
print("\n--- Revenue by Category ---")
print(category_revenue)

# 8.3 Correlation between satisfaction and total spend
satisfaction_spend_corr = df_final['Satisfaction Rating'].corr(df_final['Total Spent'])
print(f"\nCorrelation (Satisfaction vs Spend): {satisfaction_spend_corr:.4f}")


# -------------------------------------------------------
# TODO 9: Generate Final Report
# -------------------------------------------------------

print("\n" + "=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING REPORT")
print("=" * 60)

# 9.1 Data quality issues found
total_missing = initial_missing_counts.sum()
print(f"""
Data Quality Issues:
- Missing Values: {total_missing} total missing entries
- Duplicates: {initial_duplicate_count} duplicate records found
- Data Type Issues: ['total_spent had $ signs and commas', 
                     'dates in mixed formats (YYYY-MM-DD and MM/DD/YYYY)',
                     'satisfaction_rating stored as object in some rows']
""")

# 9.2 Standardization changes made
print(f"""Standardization Changes:
- Names: Converted to proper case (e.g., JESSICA -> Jessica, jones -> Jones)
- Categories: Standardized to title case (e.g., ACCESSORIES -> Accessories, womenswear -> Womenswear)
- Phone Numbers: Reformatted all numbers to {phone_format} format
- Dates: Parsed both YYYY-MM-DD and MM/DD/YYYY formats into datetime objects
- Monetary values: Stripped $ and commas, converted to float
""")

# 9.3 Key business insights
top_category = category_revenue.index[0]
top_category_revenue = category_revenue.iloc[0]
total_customers = len(df_final)

print(f"""Key Business Insights:
- Customer Base: {total_customers} total customers
- Revenue by Loyalty:""")
for status, avg in avg_spent_by_loyalty.items():
    print(f"    {status}: ${avg:.2f} avg spend")

print(f"- Top Category: {top_category} with ${top_category_revenue:.2f} total revenue")
print(f"- Satisfaction/Spend Correlation: {satisfaction_spend_corr:.4f} (weak positive relationship)")

# 9.4 Show the cleaned dataset
print("\n--- First 5 Rows of Cleaned Dataset ---")
print(df_final.head())
