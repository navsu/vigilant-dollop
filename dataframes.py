# STEP 1 : ‘Load csv data into our dataframes’

import pandas as pd
df = pd.read_csv(‘train.csv’)
df.head()

# STEP 2 : Exploratory data Analysis.(EDA)
# Category Analysis

cyberfraud_Data1=news_data
cyberfraud_Data1['CATEGORY']=news_Data1['CATEGORY'].replace({'b': 'BUISNESS', 't': 'TECHNOLOGY' , 'm' :'MEDICAL', 'e':'ENTERTAINMENT'})
ax = sns.countplot(x="CATEGORY", data=news_Data1).set_title('CATEGORY')

# STEP 3 : DATA CLEANING AND DATA PREPROCESSING
# Stop Words:
# Feature Engineering :
# POS Tagging


# Step 4: CLASSIFICATION :
# There are various methods to convert word into a vector but we will use the following methods in this .
# 1. BAG OF WORDS
# 2. TFIDF 
# 3. AVG w2v
# 4. TFIDF w2v

# MACHINE LEARNING MODELS:

# 1. Naive Bayes
#    grid search
 
# 2. Logistic Regression
#    CONFUSION MATRIX
#    FEATURE IMPORTANCE :

# Women/Child	Related	Crime : [feature_names]
# Financial	Fraud	Crimes    : [feature_names]
# Other	Cyber	Crime	        : [feature_names]

# plot the feature importance using coef attribute in Logistic Regression

# LSTM:

# Our text preprocessing will include the following steps:

# Convert all text to lower case.
# Replace REPLACE_BY_SPACE_RE symbols by space in text.
# Remove symbols that are in BAD_SYMBOLS_RE from text.
# Remove “x” in text.
# Remove stop words.

# LSTM Modeling
# Vectorize title text, by turning each text into either a sequence of integers or into a vector.
# Limit the data set to the top words.
# Set the max number of words in each complaint/category.

# Truncate and pad the input sequences so that they are all in the same length for modeling.

# Train test split.

# The first layer is the embedded layer that uses 100 length vectors to represent each word.
# The next layer is the LSTM layer with 100 memory units.
# The output layer must create n output values, one for each n class/category-subcategory.
# Activation function is softmax for multi-class classification.
# Because it is a multi-class classification problem, categorical_crossentropy is used as the loss function.



