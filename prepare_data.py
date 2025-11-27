import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
df = pd.read_csv('mtsamples.csv')
# print(f"Total file len show kro: {len(df)}")
# print(df.columns)
# print(df.head(2))
df_clean = df[['transcription', 'medical_specialty']].dropna()
print(f"Loaded {len(df_clean)} medical transcriptions") 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

chunks = []
for idx,row in df_clean.iterrows():
    splits = text_splitter.split_text(row['transcription'])
    for split in splits:
        chunks.append({
            'text': split,
            'metadata': {
                'speciality': row['medical_specialty'],
                 'doc_id': idx
            }


        }) 

print(f"Chunks we created {len(chunks)} chunks")

import pickle
with open('chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)

print("chunks are saved")


