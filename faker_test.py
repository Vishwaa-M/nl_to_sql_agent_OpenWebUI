import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from faker import Faker
import random
from datetime import date, datetime, timedelta
import numpy as np

# --- Configuration --------------------------------------------------
# !! Update with your PostgreSQL credentials !!
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "G00dne$$18"  # Replace with your password
}
DB_NAME = "agent_db"

# Number of records
NUM_PRODUCTS = 100
NUM_DISTRIBUTORS = 40
NUM_WAREHOUSES = 8
NUM_SALES_TRANSACTIONS = 3000

# --- Realistic Data for Generation ----------------------------------
fake = Faker()

THERAPEUTIC_AREAS = ["Antivirals", "Cardiovascular", "Central Nervous System", "Antibiotics", "Anti-Retrovirals", "Gastroenterological"]
DISTRIBUTOR_DATA = {
    "AmerisourceBergen": ("USA", "North America"), "McKesson": ("USA", "North America"),
    "Cardinal Health": ("USA", "North America"), "Phoenix Group": ("Germany", "Europe"),
    "Medipal Holdings": ("Japan", "Asia"), "Sinopharm": ("China", "Asia"),
    "Cencora": ("UK", "Europe"), "Uniphar": ("Ireland", "Europe"),
    "Apollo Pharmacy": ("India", "Asia"), "MedPlus": ("India", "Asia"),
    "Raia Drogasil": ("Brazil", "South America")
}

# --- Script Functions -----------------------------------------------

def create_database_if_not_exists():
    """Creates the target database if it doesn't exist."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        if not cur.fetchone():
            cur.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error checking/creating database: {e}")
        exit()

def create_tables(cur):
    """Creates five interconnected tables designed for business analysis."""
    commands = (
        """
        DROP TABLE IF EXISTS sales, inventory_snapshots, distributors, warehouses, products CASCADE;
        """,
        """
        CREATE TABLE products (
            product_id SERIAL PRIMARY KEY,
            trade_name VARCHAR(255) NOT NULL UNIQUE,
            generic_name VARCHAR(255) NOT NULL,
            therapeutic_area VARCHAR(100),
            unit_cost NUMERIC(10, 2) NOT NULL
        );
        """,
        """
        CREATE TABLE distributors (
            distributor_id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            country VARCHAR(100),
            region VARCHAR(100),
            tier VARCHAR(50) -- e.g., Gold, Silver, Bronze
        );
        """,
        """
        CREATE TABLE warehouses (
            warehouse_id SERIAL PRIMARY KEY,
            location_name VARCHAR(255) NOT NULL,
            country VARCHAR(100)
        );
        """,
        """
        CREATE TABLE inventory_snapshots (
            snapshot_id SERIAL PRIMARY KEY,
            product_id INTEGER NOT NULL REFERENCES products(product_id),
            warehouse_id INTEGER NOT NULL REFERENCES warehouses(warehouse_id),
            quantity_on_hand INTEGER NOT NULL,
            snapshot_date DATE NOT NULL,
            UNIQUE (product_id, warehouse_id, snapshot_date)
        );
        """,
        """
        CREATE TABLE sales (
            sale_id SERIAL PRIMARY KEY,
            product_id INTEGER NOT NULL REFERENCES products(product_id),
            distributor_id INTEGER NOT NULL REFERENCES distributors(distributor_id),
            warehouse_id INTEGER NOT NULL REFERENCES warehouses(warehouse_id),
            quantity_sold INTEGER NOT NULL,
            unit_price NUMERIC(10, 2) NOT NULL,
            total_revenue NUMERIC(12, 2) NOT NULL,
            total_cost NUMERIC(12, 2) NOT NULL,
            profit NUMERIC(12, 2) NOT NULL,
            sale_date DATE NOT NULL
        );
        """
    )
    for command in commands:
        cur.execute(command)
    print("All 5 tables created successfully.")

def generate_and_insert_data(cur):
    """Generates and inserts realistic, patterned data."""
    # 1. Products
    products_data = []
    for _ in range(NUM_PRODUCTS):
        area = random.choice(THERAPEUTIC_AREAS)
        g_name = fake.unique.lexify(text=f'????{random.choice(["cin","vir","zole","pril"])}', letters='abcdefghijklmnopqrstuvwxyz')
        t_name = f"{area.split(' ')[0]}-{g_name.capitalize()}"
        cost = round(random.uniform(0.50, 20.00), 2)
        products_data.append((t_name, g_name, area, cost))
    cur.executemany("INSERT INTO products (trade_name, generic_name, therapeutic_area, unit_cost) VALUES (%s, %s, %s, %s)", products_data)
    cur.execute("SELECT product_id, therapeutic_area, unit_cost FROM products;")
    products = cur.fetchall()
    print(f"-> Inserted {cur.rowcount} products.")
    
    # Designate bestsellers
    bestseller_ids = [p[0] for p in random.sample(products, int(NUM_PRODUCTS * 0.2))]

    # 2. Distributors
    dist_data = []
    names = list(DISTRIBUTOR_DATA.keys())
    for i in range(NUM_DISTRIBUTORS):
        name = names[i % len(names)] + (f" Division {i//len(names)+1}" if i >= len(names) else "")
        country, region = DISTRIBUTOR_DATA[names[i % len(names)]]
        tier = random.choices(["Gold", "Silver", "Bronze"], weights=[0.3, 0.5, 0.2], k=1)[0]
        dist_data.append((name, country, region, tier))
    cur.executemany("INSERT INTO distributors (name, country, region, tier) VALUES (%s, %s, %s, %s)", dist_data)
    cur.execute("SELECT distributor_id, country, tier FROM distributors;")
    distributors = cur.fetchall()
    print(f"-> Inserted {cur.rowcount} distributors.")

    # 3. Warehouses
    wh_data = []
    unique_countries = {d[1] for d in distributors}
    for country in unique_countries:
        for i in range(NUM_WAREHOUSES // len(unique_countries)):
             wh_data.append((f"{country} WH {i+1}", country))
    cur.executemany("INSERT INTO warehouses (location_name, country) VALUES (%s, %s)", wh_data)
    cur.execute("SELECT warehouse_id, country FROM warehouses;")
    warehouses = cur.fetchall()
    print(f"-> Inserted {cur.rowcount} warehouses.")

    # 4. Sales with realistic patterns
    sales_data = []
    for _ in range(NUM_SALES_TRANSACTIONS):
        # Prefer bestsellers and gold-tier distributors
        product_id, area, cost = random.choice(products)
        if random.random() < 0.6: # 60% chance to pick a bestseller
            product_id = random.choice(bestseller_ids)
            # Refetch details for the chosen bestseller
            prod_details = next(p for p in products if p[0] == product_id)
            area, cost = prod_details[1], prod_details[2]
            
        distributor_id, country, tier = random.choice(distributors)
        if random.random() < 0.5: # 50% chance to pick a gold distributor
            gold_distributors = [d for d in distributors if d[2] == "Gold"]
            if gold_distributors:
                distributor_id, country, tier = random.choice(gold_distributors)
        
        # Match warehouse to distributor country
        possible_whs = [wh[0] for wh in warehouses if wh[1] == country]
        if not possible_whs: continue
        warehouse_id = random.choice(possible_whs)
        
        sale_date = fake.date_between(start_date='-3y', end_date='today')
        
        # Introduce seasonality for Antivirals
        quantity = random.randint(100, 2000)
        if area == "Antivirals" and sale_date.month in [10, 11, 12, 1, 2]:
            quantity *= random.randint(2, 4) # Spike for flu season

        margin = 1.2 + (random.random() * 0.8) # 20% to 100% margin
        unit_price = round(float(cost) * margin, 2)
        
        sales_data.append((product_id, distributor_id, warehouse_id, quantity, unit_price,
                           quantity * unit_price, quantity * float(cost), 
                           (quantity * unit_price) - (quantity * float(cost)), sale_date))

    cur.executemany("INSERT INTO sales (product_id, distributor_id, warehouse_id, quantity_sold, unit_price, total_revenue, total_cost, profit, sale_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", sales_data)
    print(f"-> Inserted {cur.rowcount} sales records with business logic.")
    
    # 5. Inventory Snapshots
    snapshot_data = []
    for product in products:
        for warehouse in warehouses:
            for i in range(36): # Monthly snapshots for 3 years
                snapshot_date = date.today().replace(day=1) - timedelta(days=i*30)
                quantity_on_hand = random.randint(500, 10000)
                # Boost inventory for bestsellers
                if product[0] in bestseller_ids:
                    quantity_on_hand *= 2
                snapshot_data.append((product[0], warehouse[0], quantity_on_hand, snapshot_date))
                
    cur.executemany("INSERT INTO inventory_snapshots (product_id, warehouse_id, quantity_on_hand, snapshot_date) VALUES (%s, %s, %s, %s)", snapshot_data)
    print(f"-> Inserted {cur.rowcount} monthly inventory snapshots.")


def main():
    """Main function to orchestrate DB and data setup."""
    create_database_if_not_exists()
    
    conn = None
    try:
        db_conn_str = f"dbname='{DB_NAME}' user='{DB_CONFIG['user']}' host='{DB_CONFIG['host']}' password='{DB_CONFIG['password']}'"
        conn = psycopg2.connect(db_conn_str)
        conn.autocommit = True
        cur = conn.cursor()
        
        create_tables(cur)
        generate_and_insert_data(cur)
        
        cur.close()
        print("\nAnalyzable pharma database created and populated successfully! ✔")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    main()