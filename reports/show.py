import pandas as pd

# Read the CSV file
df = pd.read_csv('reports/metric_scores.txt')

# Format the scores to be more readable (round to 4 decimal places)
df_formatted = df.round(4)

# Display the DataFrame with better formatting
print("\nMetric Scores for Each Report:")
print("=" * 100)
for _, row in df_formatted.iterrows():
    print(f"\nStudy ID: {row['study_id']}")
    print(f"Report: {row['report']}")
    print("\nScores:")
    print(f"- BLEU Score:        {row['bleu_score']:.4f}")
    print(f"- BERT Score:        {row['bertscore']:.4f}")
    print(f"- Semantic Score:    {row['semb_score']:.4f}")
    print(f"- RadGraph Combined: {row['radgraph_combined']:.4f}")
    print(f"- RadCliQ-v0:        {row['RadCliQ-v0']:.4f}")
    print(f"- RadCliQ-v1:        {row['RadCliQ-v1']:.4f}")
    print("-" * 80)