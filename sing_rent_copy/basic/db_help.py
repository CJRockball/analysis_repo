import pandas as pd
import sqlite3

def get_data():
    conn = sqlite3.connect('basic/rental.db')
    cur = conn.cursor()
    
    db_command = """
                SELECT * FROM rental_data;
                """
    cur.execute(db_command)
    result = cur.fetchall()
    
    conn.commit()
    conn.close()
    
    return result


def get_x_data(lines):
    conn = sqlite3.connect('basic/rental.db')
    cur = conn.cursor()
    
    db_command = """
                SELECT * FROM rental_data LIMIT(?);
                """
    cur.execute(db_command, ((lines),) )
    result = cur.fetchall()
    
    conn.commit()
    conn.close()
    
    return result

def project_search(project):
    conn = sqlite3.connect('basic/rental.db')
    cur = conn.cursor()
    
    db_command = """
                SELECT * FROM rental_data WHERE project=(?);
                """
    cur.execute(db_command, ((project),))
    result = cur.fetchall()
    
    conn.commit()
    conn.close()
    return result    
    
    
