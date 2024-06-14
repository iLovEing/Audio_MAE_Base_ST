from mysql import connector


class MySQLHandler:
    def __init__(self, host, user, password, database=None):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

        self._connect()

    def _connect(self):
        try:
            self.connection = connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor()
            # print("Connected to MySQL database.")
            print("Connected to MySQL database.")
            print("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
            print("Set the transaction isolation level to 'READ COMMITTED'.")
        except connector.Error as err:
            print(f"Error: {err}")
            print(f"Error: {err}")
            self.connection = None

    def query(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    @property
    def head_dict(self):
        if self.cursor is not None:
            return {column[0]: i for i, column in enumerate(self.cursor.description)}
        else:
            print('empty handle callback')
            return None

    def insert(self, insert_state, value):
        self.cursor.execute(insert_state, value)
        self.connection.commit()

    def close_connection(self):
        if self.connection is not None:
            self.connection.close()
            print("Connection closed.")


    # def create_database(self, db_name):
    #     try:
    #         self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    #         print(f"Database '{db_name}' created successfully.")
    #     except mysql.connector.Error as err:
    #         print(f"Error: {err}")
    #
    # def create_table(self, table_name, columns):
    #     try:
    #         create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
    #         self.cursor.execute(create_table_query)
    #         print(f"Table '{table_name}' created successfully.")
    #     except mysql.connector.Error as err:
    #         print(f"Error: {err}")
    #
    # def insert_data(self, table_name, data):
    #     try:
    #         columns = ', '.join(data.keys())
    #         values = ', '.join(f"'{value}'" for value in data.values())
    #         insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
    #         self.cursor.execute(insert_query)
    #         self.connection.commit()
    #         print("Data inserted successfully.")
    #     except mysql.connector.Error as err:
    #         print(f"Error: {err}")
    #
    # def read_data(self, table_name):
    #     try:
    #         select_query = f"SELECT * FROM {table_name}"
    #         self.cursor.execute(select_query)
    #         result = self.cursor.fetchall()
    #         for row in result:
    #             print(row)
    #         return result
    #     except mysql.connector.Error as err:
    #         print(f"Error: {err}")
    #
    #
    # def get_current_database(self):
    #     try:
    #         self.cursor.execute("SELECT DATABASE()")
    #         result = self.cursor.fetchone()
    #         if result:
    #             return result[0]
    #         else:
    #             return None
    #     except mysql.connector.Error as err:
    #         print(f"Error: {err}")
    # def change_database(self, new_database):
    #     try:
    #         self.cursor.execute(f"USE {new_database}")
    #         self.database = new_database
    #         print(f"Changed to database: {new_database}")
    #     except mysql.connector.Error as err:
    #         print(f"Error: {err}")







