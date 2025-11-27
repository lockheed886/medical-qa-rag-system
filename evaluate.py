import pandas as pd
from qa_pipeline import qa_chain  # import your chain

test_questions = [
    "What are the symptoms of hypertension?",
    "How is diabetes diagnosed?",
    "What causes chest pain?",
]

results = []

for question in test_questions:
    response = qa_chain({"query": question})
    
    results.append({
        'question': question,
        'answer': response['result'],
        'num_sources': len(response['source_documents']),
        'specialties': [doc.metadata.get('specialty') for doc in response['source_documents']]
    })

eval_df = pd.DataFrame(results)
eval_df.to_csv('evaluation_results.csv', index=False)
print("Evaluation complete!")
