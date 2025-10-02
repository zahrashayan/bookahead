from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import re

# Google Flights URL
url = "https://www.google.com/travel/flights/search?tfs=CBwQAhojEgoyMDI1LTExLTIwagwIAhIIL20vMGQ2bHByBwgBEgNKRksaIxIKMjAyNS0xMS0yN2oHCAESA0pGS3IMCAISCC9tLzBkNmxwQAFIAXABggELCP___________wGYAQE&tfu=EgIIACIA"

# Chrome options
options = webdriver.ChromeOptions()
# options.add_argument("--headless=new")
# options.add_argument("--disable-gpu")
# options.add_argument("--no-sandbox")
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)

try:
    driver.get(url)
    print("Opened Google Flights... waiting for prices to load")

    # 3️⃣ Wait for all elements with aria-live="polite"
    wait = WebDriverWait(driver, 30)

    # find elements by classname : zISZ5c QB2Jof
    elements = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "zISZ5c.QB2Jof")))

    # if inner text contains "more", click
    for el in elements:
        if "more" in el.text.lower():
            print("Clicking 'more' button to load additional prices...")
            el.click()

    sleep(5)

    # wait a bit more for prices to load
    wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "pIav2d")))

    # get all elements with class name pIav2d
    elements_top = driver.find_elements(By.CLASS_NAME, "pIav2d")
    print(f"Found {len(elements_top)} elements with class name 'pIav2d'")

    for el in elements_top:
        # get elements under class name YMlIz FpEdX
        price_elements = el.find_elements(By.CLASS_NAME, "YMlIz.FpEdX")
        for sub_el in price_elements:
            price = sub_el.text.strip()
            if price.startswith("$"):
                print(f"Price: {price}")

        # Prefer airline from logo image 'alt' attribute when available
        airline = None
        try:
            imgs = el.find_elements(By.TAG_NAME, "img")
            for img in imgs:
                alt = img.get_attribute("alt")
                if alt and alt.strip():
                    airline = alt.strip()
                    break
        except Exception:
            airline = None

        # Fallback: textual airline candidates, but filter out times/routes/durations
        if not airline:
            airline_elements = el.find_elements(By.CLASS_NAME, "sSHqwe.tPgKwe.ogfYpf")
            for a in airline_elements:
                text = a.text.strip()
                if not text:
                    continue
                # Filter heuristics
                # times like 6:29 AM, 06:29, or with +1
                if re.search(r"\d{1,2}:\d{2}(?:\s*[APMapm\.]{2,4})?(?:\+\d)?", text):
                    continue
                # routes like SFO–JFK or SFO - JFK or 'to'
                if '–' in text or '—' in text or '->' in text or ' to ' in text.lower() or re.search(r"[A-Z]{3}[-–—][A-Z]{3}", text):
                    continue
                # durations like '1 hr 54 min' or '54 min'
                if re.search(r"\d+\s*(?:h|hr|hour|min)", text.lower()):
                    continue
                # strings composed mostly of digits/punctuation
                if re.fullmatch(r"[\d\s:–—+\-]+", text):
                    continue
                # likely airline name
                airline = text
                break

        if airline:
            print(f"Airline: {airline}")

        time_elements = el.find_elements(By.CLASS_NAME, "zxVSec.YMlIz.tPgKwe.ogfYpf")
        for idx, t in enumerate(time_elements):
            time_text = t.text.strip()
            if not time_text:
                continue
            if idx == 0:
                print(f"Departure: {time_text}")
            elif idx == 1:
                print(f"Arrival: {time_text}")

        stop_elements = el.find_elements(By.CLASS_NAME, "EfT7Ae.AdWm1c.tPgKwe")
        for s in stop_elements:
            stop_text = s.text.strip()
            if stop_text:
                print(f"Stops: {stop_text}")

    # 4️⃣ Look for first element that looks like a price
    price_text = None
    for el in elements:
        text = el.text.strip()
        if text.startswith("from $"):
            price_text = text
            break

    # 5️⃣ Print result
    if price_text:
        clean_price = int(price_text.replace("from $", "").replace(",", ""))
        print(f"✅ Flight price found: {clean_price}")
    else:
        print("❌ No price element matched 'from $...'")

# YMlIz FpEdX jLMuyc --> green
# YMlIz FpEdX --> normal
# button --> zISZ5c QB2Jof

except Exception as e:
    print("❌ Error while scraping:", e)

# finally:
#     driver.quit()

# keep window open
input("Press Enter to close the browser and exit...")



