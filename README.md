# BookAhead

BookAhead predicts **when is the best time to book a flight** for the lowest price.

It scrapes real data from Google Flights a few times a day for selected routes  
(SFO → NYC, SFO → ISB, and SFO → SAN). Each run logs flight prices for multiple departure dates at different hours (morning, noon, evening, late night).  
The data is cleaned, processed, and used to train small route-specific machine learning models.

It is important because flight prices are not random — they follow patterns based on time, weekday, and proximity to travel.
By combining scraping with ML, BookAhead aims to forecast the best day to buy your ticket, instead of relying on guesswork.
---

## What this project does

1. **Scrape live flight prices**  
   Using Selenium, it pulls airline, price, stops, and flight times from Google Flights.

2. **Clean + structure data**  
   Converts raw scraped data into a tidy dataset that includes:
   - days until departure  
   - weekday/weekend patterns  
   - time-of-day effects

3. **Train predictive models**  
   Currently, a Random Forest Regressor is trained **per route** to learn price behavior over time (will expand)

---

## How to run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn pyarrow selenium joblib matplotlib

2. To run the scraper 
python src/scrapers/scraper_v2.py

3. To clean and prepare
python src/ml/prepare_panel.py

4. To train the model
python src/ml/pipeline.py

## Example Output 
==================== SFO-NYC ====================
MAE:  $20.01
RMSE: $21.18
R²:   -0.004
✓ Saved model for SFO-NYC → models/rf_SFO_NYC.pkl
