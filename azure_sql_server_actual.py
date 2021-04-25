import base64
import io
import textwrap
import numpy as np
import cv2
import pyodbc
import datetime


class Database:
    is_connection = False

    def open_connection(self):
        if not Database.is_connection:
            driver = '{ODBC Driver 17 for SQL Server}'
            server_name = 'mycovidkeeper'
            database_name = 'mySampleDatabase'
            server = '{server_name}.database.windows.net'.format(server_name=server_name)
            username = 'azureuser'
            password = 'Amitay5925'

            connection_string = textwrap.dedent(f'''
                Driver={driver};
                Server={server};
                Database={database_name};
                Uid={username};
                Pwd={password};
                Encrypt=yes;
                TrustServerCertification=no;
                Connection Timeout=30;
            ''')

            self.cnxn: pyodbc.Connection = pyodbc.connect(connection_string)
            Database.is_connection = True

    def close_connection(self):
        self.cnxn.close()
        Database.is_connection = False

    def open_cursor(self):
        self.crsr: pyodbc.Cursor = self.cnxn.cursor()

    def close_cursor(self):
        self.crsr.close()

    def convert_image_to_varbinary(self, filename):
        image = open(filename, 'rb')
        image_read = image.read()
        image_64_encode = base64.encodebytes(image_read)
        image.close()
        return image_64_encode

    def insert_worker(self):
        self.open_connection()
        self.open_cursor()
        insert_sql = "INSERT INTO [dbo].[Workers] (Id, FullName, Email_address, Image) " \
                     "VALUES (?,?,?,?)"
        image = self.convert_image_to_varbinary('Faces/Amitai1.jpg')
        self.crsr.execute(insert_sql, 2, 'Amitai Popovsky', 'amitai.p2@gmail.com', image)
        # image = self.convert_image_to_varbinary('Faces/Aberdam.jpg')
        # self.crsr.execute(insert_sql, 1, 'Nerya Aberdam', 'naberdam@gmail.com', image)
        self.crsr.commit()
        self.close_cursor()

    def get_image_worker(self, id_worker):
        self.open_connection()
        self.open_cursor()
        select_sql = "SELECT Image From [dbo].[Workers] Where Id=" + id_worker
        # image = self.convert_image_to_varbinary('Faces/Aberdam.jpg')
        self.crsr.execute(select_sql)
        data = self.crsr.fetchone()[0]
        data = self.convert_bytes_to_image(data)
        self.crsr.commit()
        self.close_cursor()

    def convert_bytes_to_image(self, data):
        stream = io.BytesIO(data)
        _stream = stream.getvalue()
        image = cv2.imdecode(np.fromstring(_stream, dtype=np.uint8), 1)
        # cv2.imshow("Faces found", image)
        # cv2.waitKey(0)
        return image

    def get_workers_to_images_dict(self):
        self.open_connection()
        self.open_cursor()
        select_sql = "SELECT Id, Image " \
                     "FROM [dbo].[Workers]"
        self.crsr.execute(select_sql)
        # data = self.crsr.fetchone()
        workers_to_images_dict = {}
        for details in self.crsr:
            image = self.convert_bytes_to_image(details[1])
            workers_to_images_dict[details[0]] = image
        self.close_cursor()
        # self.crsr.commit()
        return workers_to_images_dict

    def get_fullname_and_email_by_id(self, id_worker):
        select_sql = "SELECT FullName, Email_address From [dbo].[Workers] Where Id=" + id_worker
        result = self.select_query_of_one_row(select_sql)
        fullname = result[0]
        email = result[1]
        return fullname, email

    def insert_event(self, id_worker):
        insert_sql = "INSERT INTO [dbo].[History_events] (Id_worker, Time_of_event) " \
                     "VALUES (?,?)"
        f = '%Y-%m-%d %H:%M:%S'
        values_list = [id_worker, datetime.datetime.now().strftime(f)]
        self.insert_query_of_one_row(query=insert_sql, values_list=values_list)

    def get_events_order_with_max_time(self):
        # self.open_connection()
        # self.open_cursor()
        # self.crsr.execute(select_sql)
        # # data = self.crsr.fetchone()
        # events_order_by_max_time = []
        # for details in self.crsr:
        #     events_order_by_max_time.append(details)
        # self.close_cursor()
        # self.crsr.commit()
        select_sql = "select id_worker, Max(Time_of_event) from [dbo].[History_events] " \
                     "group by id_worker order by Max(Time_of_event) desc;"
        result = self.select_query_of_many_rows(select_sql)
        events_order_by_max_time = []
        for details in result:
            events_order_by_max_time.append(details)
        return events_order_by_max_time

    def get_max_time_of_event_by_id_worker(self, id_worker):
        result = self.select_query_of_one_row("select Max(Time_of_event) "
                                              "from [dbo].[History_events] where Id_worker=" + id_worker)
        if not result:
            return None
        return result[0]

    def select_query_of_one_row(self, query):
        self.open_connection()
        self.open_cursor()
        select_sql = query
        self.crsr.execute(select_sql)
        result = self.crsr.fetchone()
        self.close_cursor()
        return result

    def select_query_of_many_rows(self, query):
        self.open_connection()
        self.open_cursor()
        select_sql = query
        self.crsr.execute(select_sql)
        result = self.crsr.fetchall()
        self.close_cursor()
        return result

    def insert_query_of_one_row(self, query, values_list):
        self.open_connection()
        self.open_cursor()
        insert_sql = query
        self.crsr.execute(insert_sql, values_list)
        self.crsr.commit()
        self.close_cursor()

# b = Database()
# print(b.get_events_order_with_max_time())
# print(b.get_max_time_of_event_by_id_worker('2'))
# b.insert_event('2')
# data1, m = b.get_fullname_and_email_by_id('1')
# print(data1)
# print(m)
# print(b.get_workers_to_images_dict())
# b.insert_worker()
# b.get_image_worker('4')
# select_sql = "SELECT * From [SalesLT].[Address]"
# results = crsr.execute(select_sql)
# print(crsr.fetchall())


# cnxn.close()
