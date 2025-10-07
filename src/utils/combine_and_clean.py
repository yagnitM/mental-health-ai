import os
import pandas as pd
import re
from datetime import datetime
import pytz

DATA_DIR = '../../data/raw'
OUTPUT_FILE = '../../data/processed/cleaned_combined_data.csv'
MIN_TEXT_LENGTH = 30

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
combined_df = pd.DataFrame()

for file in csv_files:
    filepath = os.path.join(DATA_DIR, file)
    try:
        df = pd.read_csv(filepath)
        df['label'] = file.replace('.csv', '').lower()
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    except Exception as e:
        print(f"Error loading {file}: {e}")

print(f"Original shape: {combined_df.shape}")

def convert_utc_to_ist(utc_timestamp):
    if pd.isna(utc_timestamp):
        return None
    try:
        utc_time = datetime.utcfromtimestamp(utc_timestamp)
        ist = pytz.timezone('Asia/Kolkata')
        datetime_ist = utc_time.replace(tzinfo=pytz.utc).astimezone(ist)
        return datetime_ist.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return None

if 'created_utc' in combined_df.columns:
    combined_df['created_ist'] = combined_df['created_utc'].apply(convert_utc_to_ist)

columns_to_drop = ['id', 'score', 'num_comments', 'created_utc']  
columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
combined_df.drop(columns=columns_to_drop, inplace=True)

combined_df.drop_duplicates(subset=['title', 'text'], inplace=True)
combined_df.dropna(subset=['title', 'text'], inplace=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) 
    text = re.sub(r'/u/\w+|/r/\w+', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    return text

combined_df['title'] = combined_df['title'].apply(clean_text)
combined_df['text'] = combined_df['text'].apply(clean_text)

combined_df['combined_text'] = combined_df['title'] + ' ' + combined_df['text']
combined_df['combined_text'] = combined_df['combined_text'].apply(clean_text)

combined_df = combined_df[combined_df['combined_text'].str.len() > MIN_TEXT_LENGTH]
combined_df = combined_df[combined_df['combined_text'].str.contains(r'[a-zA-Z]', regex=True)]
combined_df.dropna(inplace=True)
combined_df.reset_index(drop=True, inplace=True)

print("\nClass distribution:")
print(combined_df['label'].value_counts())

if 'created_ist' in combined_df.columns:
    print(f"\nTemporal range (IST):")
    print(f"Earliest post: {combined_df['created_ist'].min()}")
    print(f"Latest post: {combined_df['created_ist'].max()}")

combined_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nCleaned data saved to: {OUTPUT_FILE}")
print(f"Final shape: {combined_df.shape}")
print(f"Final columns: {list(combined_df.columns)}")