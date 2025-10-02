from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import csv
from datetime import datetime
import os

# Google Flights URL
url = "https://www.google.com/travel/flights/search?tfs=CBwQAhojEgoyMDI1LTExLTIwagwIAhIIL20vMGQ2bHByBwgBEgNKRksaIxIKMjAyNS0xMS0yN2oHCAESA0pGS3IMCAISCC9tLzBkNmxwQAFIAXABggELCP___________wGYAQE&tfu=EgIIACIA"

# Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)

# List to store all flight data
flights = []

# CSV filename based on today's date only
csv_filename = f'flights_{datetime.now().strftime("%Y%m%d")}.csv'

try:
    driver.get(url)
    print("Opened Google Flights... waiting for prices to load")
    
    # Wait for page to load
    wait = WebDriverWait(driver, 30)
    
    try:
        # Wait for initial content to load
        wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "pIav2d")))
        sleep(2)
        
        # Look for "more" buttons
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, ".zISZ5c.QB2Jof")
            for el in elements:
                if "more" in el.text.lower():
                    print("Clicking 'more' button to load additional prices...")
                    el.click()
                    sleep(3)
        except Exception as e:
            print(f"No 'more' button found or couldn't click: {e}")
    
    except Exception as e:
        print(f"Initial load error: {e}")
    
    # Get all flight result containers
    elements_top = driver.find_elements(By.CLASS_NAME, "pIav2d")
    print(f"Found {len(elements_top)} flight results\n")
    
    flight_count = 0
    for idx, el in enumerate(elements_top, 1):
        # Skip empty results
        if not el.text.strip():
            continue
        
        flight_count += 1
        print(f"--- Flight {flight_count} ---")
        
        # Initialize flight data dictionary
        flight_data = {
            'flight_number': flight_count,
            'price': '',
            'airline': '',
            'departure_time': '',
            'arrival_time': '',
            'stops': '',
            'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Get price
        try:
            price_elements = el.find_elements(By.CSS_SELECTOR, ".YMlIz.FpEdX")
            for sub_el in price_elements:
                text = sub_el.text.strip()
                if text and "$" in text:
                    price = text.replace("$", "").replace(",", "")
                    flight_data['price'] = price
                    print(f"Price: ${price}")
                    break
        except Exception as e:
            print(f"Price error: {e}")
        
        # Get airline (only first one - actual airline name)
        try:
            airline_elements = el.find_elements(By.CSS_SELECTOR, ".sSHqwe.tPgKwe.ogfYpf")
            if airline_elements:
                airline = airline_elements[0].text.strip()
                flight_data['airline'] = airline
                print(f"Airline: {airline}")
        except Exception as e:
            print(f"Airline error: {e}")
        
        # Get times
        try:
            time_elements = el.find_elements(By.CSS_SELECTOR, ".zxVSec.YMlIz.tPgKwe.ogfYpf")
            times = [t.text.strip() for t in time_elements if t.text.strip()]
            
            if times:
                time_str = " ".join(times)
                if "–" in time_str:
                    parts = time_str.split("–")
                    departure = parts[0].strip()
                    arrival = parts[1].strip()
                    flight_data['departure_time'] = departure
                    flight_data['arrival_time'] = arrival
                    print(f"Departure: {departure}")
                    print(f"Arrival: {arrival}")
                else:
                    flight_data['departure_time'] = time_str
                    print(f"Time: {time_str}")
        except Exception as e:
            print(f"Time error: {e}")
        
        # Get stops
        try:
            stop_elements = el.find_elements(By.CSS_SELECTOR, ".EfT7Ae.AdWm1c.tPgKwe")
            for s in stop_elements:
                stop_text = s.text.strip()
                if stop_text:
                    flight_data['stops'] = stop_text
                    print(f"Stops: {stop_text}")
        except Exception as e:
            print(f"Stops error: {e}")
        
        # Add flight to list
        flights.append(flight_data)
        print()  # Blank line between flights
    
    # Append to CSV (or create if doesn't exist)
    if flights:
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['flight_number', 'price', 'airline', 'departure_time', 
                         'arrival_time', 'stops', 'scraped_at']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Only write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(flights)
        
        if file_exists:
            print(f"\n✅ Appended {len(flights)} flights to {csv_filename}")
        else:
            print(f"\n✅ Created {csv_filename} with {len(flights)} flights")
        
        print(f"📊 This session's price range: ${min(f['price'] for f in flights if f['price'])} - ${max(f['price'] for f in flights if f['price'])}")
    else:
        print("\n❌ No flights found to save")

except Exception as e:
    print(f"❌ Error while scraping: {e}")
    import traceback
    traceback.print_exc()

# Keep window open
input("Press Enter to close the browser and exit...")
driver.quit()