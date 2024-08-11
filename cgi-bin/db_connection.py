import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="skocsis2",
        password="spCk7725!",
        database="multiomics"
    )
