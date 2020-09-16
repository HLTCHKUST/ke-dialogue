import os, sys
import pandas as pd
import numpy as np

###
# Filter Dataframe function
###
# Function to filter columns from knowledge-base dataframe with a given list of fields and drop all duplicated rows
def filter_columns_from_kb(kb, fields):
    filtered_kb = kb.loc[:,fields]
    return filtered_kb.drop_duplicates().reset_index(drop=True)

# Function to query equal from knowledge-base dataframe with a given list of filter tuple (field, value)
def query_equal_from_kb(kb, filters):
    filtered_kb = kb
    for field, value in filters:
        filtered_kb = filtered_kb[filtered_kb[field] == value]
    return filtered_kb.reset_index(drop=True)

# Function to query not equal from knowledge-base dataframe with a given list of filter tuple (field, value)
def query_unequal_from_kb(kb, filters):
    filtered_kb = kb
    for field, value in filters:
        filtered_kb = filtered_kb[filtered_kb[field] != value]
    return filtered_kb.reset_index(drop=True)   