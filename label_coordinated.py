import csv
import re
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Pastikan file tokopedia_product_reviews_2025.csv ada di folder yang sama
INPUT_FILE = 'tokopedia_product_reviews_2025.csv'
OUTPUT_FILE = 'tokopedia_labeled.csv'

with open(INPUT_FILE, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} reviews")

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

for r in rows:
    r['clean_text'] = clean_text(r['review_text'])
    r['date_obj'] = datetime.strptime(r['review_date'], '%Y-%m-%d')
    r['coordinated'] = 0  # default label

shop_reviews = defaultdict(list)
for r in rows:
    shop_reviews[r['shop_id']].append(r)

# A group of reviews is "coordinated" if:
# - Same shop
# - Posted within 48-hour window
# - At least 3 reviews in that window
# - All same rating AND avg cosine similarity >= 0.7

TIME_WINDOW = timedelta(hours=48)
MIN_GROUP_SIZE = 3
SIMILARITY_THRESHOLD = 0.7

coordinated_count = 0

for shop_id, reviews in shop_reviews.items():
    reviews.sort(key=lambda x: x['date_obj'])

    for i in range(len(reviews)):
        group = [reviews[i]]
        for j in range(i + 1, len(reviews)):
            if reviews[j]['date_obj'] - reviews[i]['date_obj'] <= TIME_WINDOW:
                group.append(reviews[j])
            else:
                break

        if len(group) < MIN_GROUP_SIZE:
            continue

        ratings = [r['rating'] for r in group]
        all_same_rating = len(set(ratings)) == 1

        texts = [r['clean_text'] for r in group]
        if len(set(texts)) == 1:
            avg_sim = 1.0
        else:
            try:
                tfidf = TfidfVectorizer(min_df=1).fit_transform(texts)
                sims = cosine_similarity(tfidf)
                upper = sims[np.triu_indices(len(texts), k=1)]
                avg_sim = float(np.mean(upper)) if len(upper) > 0 else 0.0
            except:
                avg_sim = 0.0

        if all_same_rating and avg_sim >= SIMILARITY_THRESHOLD:
            for r in group:
                r['coordinated'] = 1
                coordinated_count += 1

print(f"Labeled {coordinated_count} reviews as coordinated (before dedup)")

labeled_ids = set(r['review_id'] for r in rows if r['coordinated'] == 1)
print(f"Unique coordinated reviews : {len(labeled_ids)}")
print(f"Normal reviews             : {len(rows) - len(labeled_ids)}")
print(f"Coordinated ratio          : {len(labeled_ids)/len(rows)*100:.2f}%")

fieldnames = list(rows[0].keys())
for field in ['clean_text', 'date_obj']:
    if field in fieldnames:
        fieldnames.remove(field)

with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        row_out = {k: v for k, v in r.items() if k in fieldnames}
        writer.writerow(row_out)

print(f"\nDone! Saved to {OUTPUT_FILE}")
