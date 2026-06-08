# ==================================================
# MOCK DATA FOR STREAMLIT DASHBOARD
# ==================================================

# Pandas -> Used for tables and datasets
import pandas as pd

# Numpy -> Used to generate random sample values
import numpy as np


# ==================================================
# ACTIVITY OVERVIEW DATA
# ==================================================
# Used in:
# Case Insights Dashboard
#
# Creates 30 days of sample activity
# showing number of queries and documents
# processed each day.
# ==================================================

def generate_activity_data():

    dates = pd.date_range(
        start="2024-01-01",
        periods=30
    )

    return pd.DataFrame({
        "Date": dates,

        # Random query volume
        "Queries": np.random.randint(
            300,
            700,
            30
        ),

        # Random documents processed
        "Documents": np.random.randint(
            50,
            250,
            30
        )
    })


# ==================================================
# CASE STATUS DATA
# ==================================================
# Used for Pie Chart
#
# Open
# Closed
# Pending
# ==================================================

case_status_df = pd.DataFrame({

    "Status": [
        "Open",
        "Closed",
        "Pending"
    ],

    "Count": [
        230,
        980,
        242
    ]
})


# ==================================================
# QUERY CATEGORY DATA
# ==================================================
# Used in:
# RAG Analytics Bar Chart
# ==================================================

query_category_df = pd.DataFrame({

    "Category": [
        "Core Law",
        "Statutes",
        "Regulations"
    ],

    "Count": [
        350,
        280,
        190
    ]
})


# ==================================================
# SCATTER PLOT DATA
# ==================================================
# Used in:
# Query Length vs Helpfulness
#
# X-axis -> Query Length
# Y-axis -> Helpfulness Score
# ==================================================

scatter_df = pd.DataFrame({

    "Query Length": np.random.randint(
        1,
        15,
        100
    ),

    "Helpfulness": np.random.randint(
        60,
        100,
        100
    )
})


# ==================================================
# KNOWLEDGE BASE COMPOSITION
# ==================================================
# Used for Donut Chart
#
# Shows distribution of documents
# ==================================================

kb_df = pd.DataFrame({

    "Type": [
        "Case Law",
        "Briefs",
        "Regulations"
    ],

    "Count": [
        700,
        400,
        352
    ]
})


# ==================================================
# MOST ACCESSED DOCUMENTS
# ==================================================
# Used in:
# Knowledge Base Dashboard
#
# Shows documents that users
# access most frequently.
# ==================================================

accessed_docs = pd.DataFrame({

    "Document ID": [
        "DOC-001",
        "DOC-002",
        "DOC-003",
        "DOC-004",
        "DOC-005"
    ],

    "Matches": [
        425,
        389,
        310,
        288,
        250
    ]
})