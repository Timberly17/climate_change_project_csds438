import pandas as pd
import matplotlib.pyplot as plt
import os

# Load datasets
historical_data = pd.read_csv("../climate_data.csv")
future_data = pd.read_csv("../climatedata_2070.csv")

# Create figures folder if it does not exist
if not os.path.exists("figures"):
    os.mkdir("figures")

# Get columns that exist in BOTH datasets
common_columns = []

for col in historical_data.columns:
    if col in future_data.columns:
        common_columns.append(col)

# Store results
results = []

# Go through each column
for column in common_columns:

    # Only work with numeric data
    if pd.api.types.is_numeric_dtype(historical_data[column]) and pd.api.types.is_numeric_dtype(future_data[column]):

        historical_average = historical_data[column].mean()
        future_average = future_data[column].mean()

        change = future_average - historical_average

        if historical_average != 0:
            percent_change = (change / historical_average) * 100
        else:
            percent_change = 0

        results.append({
            "Indicator": column,
            "Historical": historical_average,
            "2070": future_average,
            "Change": change,
            "% Change": percent_change
        })

# Convert to table
summary_table = pd.DataFrame(results)

# Sort by biggest percent change (like your original version)
summary_table = summary_table.sort_values(by="% Change", ascending=False)

# Save results
summary_table.to_csv("engineering_summary.csv", index=False)

# Print results
print(summary_table)

# Create graphs for top 5 most changed indicators
top_indicators = summary_table.head(5)["Indicator"]

for column in top_indicators:
    historical_average = historical_data[column].mean()
    future_average = future_data[column].mean()

    plt.figure()
    plt.bar(["Historical", "2070"], [historical_average, future_average])
    plt.title("Change in " + column)
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig("figures/" + column + ".png")
    plt.close()

print("The analysis is complete. Please check the figures folder for the summary and graphs.")