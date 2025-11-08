# src/scrapers/scraper_v2.py
# uses selenium to scrape flight prices and save to a csv file

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import csv
from datetime import datetime
import os

# list of routes + urls to scrape (in flight_config.py)
from flight_config import FLIGHTS_TO_TRACK

# setup chrome driver
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.page_load_strategy = 'normal'

driver = webdriver.Chrome(options=options)
driver.set_page_load_timeout(180)  # avoid hanging forever on slow pages

# make sure file path exists for output csv
RAW_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
os.makedirs(RAW_DIR, exist_ok=True)
csv_filename = os.path.join(RAW_DIR, 'flights_data.csv')
print("Saving scraped data to:", os.path.abspath(csv_filename))

# collect all results here
all_flights = []

print(f"\n{'='*60}")
print(f"Starting scraper at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

try:
    # loop through each flight route/date config
    for idx, flight_config in enumerate(FLIGHTS_TO_TRACK, 1):
        route = flight_config['route']
        departure_date = flight_config['departure_date']
        url = flight_config['url']
        
        print(f"[{idx}/{len(FLIGHTS_TO_TRACK)}] Scraping {route} for {departure_date}...")
        
        # compute days until departure (for metadata)
        departure_dt = datetime.strptime(departure_date, '%Y-%m-%d')
        today = datetime.now()
        days_until_departure = (departure_dt - today).days
        
        # weekday info (used later as a feature)
        day_of_week = departure_dt.strftime('%A')
        is_weekend = departure_dt.weekday() >= 5
        
        # try loading the page safely
        try:
            driver.get(url)
        except Exception as e:
            print(f"   ⚠️ Failed to load page: {e}")
            print(f"   Skipping this route...\n")
            continue
        
        wait = WebDriverWait(driver, 30)
        
        try:
            # wait until flight cards appear
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "pIav2d")))
            sleep(2)
            
            # try to click "more" if button exists (sometimes flights are hidden)
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, ".zISZ5c.QB2Jof")
                for el in elements:
                    if "more" in el.text.lower():
                        el.click()
                        sleep(3)
                        break
            except:
                pass
        
        except Exception as e:
            print(f"   Error loading page: {e}")
            continue
        
        # collect all visible flight cards
        flight_elements = driver.find_elements(By.CLASS_NAME, "pIav2d")
        
        if not flight_elements:
            print(f"   No flights found")
            continue
        
        flights_found = 0
        prices = []
        
        # loop through each flight result
        for el in flight_elements:
            try:
                if not el.text.strip():
                    continue
            except:
                continue
            
            flights_found += 1
            
            # store flight info in a dict (will go to CSV)
            flight_data = {
                'scraped_date': today.strftime('%Y-%m-%d'),
                'scraped_time': today.strftime('%H:%M:%S'),
                'route': route,
                'departure_date': departure_date,
                'days_until_departure': days_until_departure,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'price': '',
                'airline': '',
                'departure_time': '',
                'arrival_time': '',
                'stops': ''
            }
            
            # extract price
            try:
                price_elements = el.find_elements(By.CSS_SELECTOR, ".YMlIz.FpEdX")
                for sub_el in price_elements:
                    text = sub_el.text.strip()
                    if text and "$" in text:
                        price = text.replace("$", "").replace(",", "")
                        flight_data['price'] = price
                        prices.append(float(price))
                        break
            except:
                pass
            
            # extract airline name
            try:
                airline_elements = el.find_elements(By.CSS_SELECTOR, ".sSHqwe.tPgKwe.ogfYpf")
                if airline_elements:
                    flight_data['airline'] = airline_elements[0].text.strip()
            except:
                pass
            
            # extract departure/arrival times
            try:
                time_elements = el.find_elements(By.CSS_SELECTOR, ".zxVSec.YMlIz.tPgKwe.ogfYpf")
                times = [t.text.strip() for t in time_elements if t.text.strip()]
                if times:
                    time_str = " ".join(times)
                    if "–" in time_str:
                        parts = time_str.split("–")
                        flight_data['departure_time'] = parts[0].strip()
                        flight_data['arrival_time'] = parts[1].strip()
                    else:
                        flight_data['departure_time'] = time_str
            except:
                pass
            
            # extract stop info (e.g. nonstop / 1 stop)
            try:
                stop_elements = el.find_elements(By.CSS_SELECTOR, ".EfT7Ae.AdWm1c.tPgKwe")
                for s in stop_elements:
                    stop_text = s.text.strip()
                    if stop_text:
                        flight_data['stops'] = stop_text
                        break
            except:
                pass
            
            # add record to master list
            all_flights.append(flight_data)
        
        # quick summary for that route/date scrape
        if prices:
            print(f"   ✓ Found {flights_found} flights")
            print(f"   Price range: ${min(prices):.0f} - ${max(prices):.0f}")
        else:
            print(f"   Found {flights_found} flights but no prices")
        
        print()
        sleep(2)

except Exception as e:
    print(f"\n❌ Critical error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # write everything to flights_data.csv (append mode)
    if all_flights:
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['scraped_date', 'scraped_time', 'route', 'departure_date', 
                         'days_until_departure', 'day_of_week', 'is_weekend', 
                         'price', 'airline', 'departure_time', 'arrival_time', 'stops']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_flights)
        
        print(f"{'='*60}")
        print(f"✅ SUCCESS!")
        print(f"   Saved {len(all_flights)} total flights to {csv_filename}")
        all_prices = [float(f['price']) for f in all_flights if f['price']]
        if all_prices:
            print(f"   Overall price range: ${min(all_prices):.0f} - ${max(all_prices):.0f}")
        print(f"{'='*60}\n")
    else:
        print(f"\n❌ No flights found to save\n")
    
    input("Press Enter to close the browser...")
    driver.quit()
