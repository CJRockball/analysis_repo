# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:55:15 2021

@author: PatCa

Setting up a basic sqlite db for rental prices in singapore
"""
import sqlite3
import numpy as np
import pandas as pd


def make_db():
    #Connect to db
    conn = sqlite3.connect("basic/rental.db")
    conn.execute("PRAGMA foreign_keys = 1")
    cur = conn.cursor()
    #Create a table
    cur.execute("""DROP TABLE IF EXISTS rental_data""")

    cur.execute("""CREATE TABLE IF NOT EXISTS rental_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER,
                    month TEXT,
                    project TEXT,
                    road TEXT,
                    district INTEGER,
                    bedrooms INTEGER,
                    sqft INTEGER,
                    price INTEGER,
                    price_sqft REAL,
                    date REAL);""")
    
    
    # Write changes
    conn.commit()
    conn.close()
    return

def load_db(df):
    conn = sqlite3.connect('basic/rental.db')
    df.to_sql('rental_data', con=conn, if_exists='append', index=False)
    conn.close()
    

if __name__ == "__main__":  
    make_db()
    #populate_db(100)
    df = pd.read_parquet('../rental_price/pp_Data/18woodsville.parquet')
    load_db(df)
    print('*** DONE ***')












    