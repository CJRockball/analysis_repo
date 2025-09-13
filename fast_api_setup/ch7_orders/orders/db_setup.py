import sqlite3
import pandas as pd
import pathlib



def setup_db(path):
    # Connect to db
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    # Create a table
    cur.execute("""DROP TABLE IF EXISTS order""")
    cur.execute("""DROP TABLE IF EXISTS order_item""")
        
    cur.execute(
        """CREATE TABLE IF NOT EXISTS order (
                    id TEXT PRIMARY KEY,
                    item TEXT,
                    status TEXT,
                    created TEXT,
                    schedule_id TEXT,
                    delivery_id TEXT);"""
    )
    
    cur.execute(
        """CREATE TABLE IF NOT EXISTS order_item (
                    id TEXT PRIMARY KEY,
                    order_id TEXT,
                    product TEXT,
                    size TEXT,
                    quantity INTEGER);"""
    )

    # Write changes
    conn.commit()
    conn.close()
    
    return


if __name__ == "__main__":
    DB_PATH = 'C:\\Users\\PatCa\\Documents\\PythonScripts\\fast_api_setup\\test_serv\\products.db'
    setup_db(DB_PATH)
    print('>>>> setup_db done')
