import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path, output_file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Check for null values and handle them if necessary
    data.fillna(method='ffill', inplace=True)  # Forward fill for missing values

    # Encode categorical variables if present (example: using LabelEncoder)
    if 'some_categorical_column' in data.columns:  # Replace with actual categorical column name
        le = LabelEncoder()
        data['some_categorical_column'] = le.fit_transform(data['some_categorical_column'])

    # Define features and target variable
    X = data[['height_left', 'height_right', 'margin_low', 'margin_up', 'length']]
    y = data['is_genuine']

    # Save the preprocessed features and target variable to a new CSV file
    preprocessed_data = pd.concat([X, y], axis=1)  # Combine features and target
    preprocessed_data.to_csv(output_file_path, index=False)  # Save to CSV

    return X, y

# Example usage
if __name__ == "__main__":
    preprocess_data('data/fake_bills.csv', 'data/preprocessed_fake_bills.csv')

