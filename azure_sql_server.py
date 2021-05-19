import base64
import io
import textwrap
import numpy as np
import cv2
import pyodbc
import datetime
from azure.storage.blob import BlobClient
# from azure_sql_server import *
import PIL.Image as Image


class Database:
    is_connection = False
    _CONNECTION_STRING = 'DefaultEndpointsProtocol=https;' \
                         'AccountName=faceimages2;' \
                         'AccountKey=vlaKfbwxn8eU1kZGo3KjuFIsgQ0BGot1MRCvs6x0mB923Yx2FOXv4XQ82Hgi' \
                         '/l4iKb4iM/DSNcAeezmYYxxFxw==;' \
                         'EndpointSuffix=core.windows.net'
    _CONTAINER = 'pictures'

    def open_connection(self):
        if not Database.is_connection:
            driver = '{ODBC Driver 17 for SQL Server}'
            server_name = 'mysqlservercovid'
            database_name = 'myCovidKeeper'
            server = '{server_name}.database.windows.net'.format(server_name=server_name)
            username = 'azureuser'
            password = 'Amitai5925'

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

    # def get_image_worker(self, id_worker):
    #     self.open_connection()
    #     self.open_cursor()
    #     select_sql = "SELECT Image From [dbo].[Workers] Where Id=" + id_worker
    #     # image = self.convert_image_to_varbinary('Faces/Aberdam.jpg')
    #     self.crsr.execute(select_sql)
    #     data = self.crsr.fetchone()[0]
    #     data = self.convert_bytes_to_image(data)
    #     self.crsr.commit()
    #     self.close_cursor()

    def start_or_close_threads(self):
        result = self.select_query_of_one_row("select Handle from [dbo].[Starter]")
        if not result:
            return None
        return result[0]

    def update_query(self, query):
        self.open_connection()
        self.open_cursor()
        self.crsr.execute(query)
        self.crsr.commit()
        self.close_cursor()

    def change_handle_value(self, value):
        if value == 0:
            self.update_query("update [dbo].[Starter] set Handle = 0")
        elif value == 1:
            self.update_query("update [dbo].[Starter] set Handle = 1")

    def fetch_photo(self, user_id: str) -> bytes:
        blob_client = BlobClient.from_connection_string(conn_str=self._CONNECTION_STRING, container_name=self.
                                                        _CONTAINER, blob_name=self._generate_blob_name(user_id))
        blob_stream = blob_client.download_blob()

        return blob_stream.readall()

    @staticmethod
    def _generate_blob_name(user_id):
        return f'{user_id}.jpg'

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
        select_sql = "SELECT Id " \
                     "FROM [dbo].[Workers]"
        self.crsr.execute(select_sql)
        # data = self.crsr.fetchone()
        workers_to_images_dict = {}
        for details in self.crsr:
            image_bytes = self.fetch_photo(details[0])
            image = self.convert_bytes_to_image(image_bytes)
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

    def get_manager_config_flag(self):
        result = self.select_query_of_one_row("select Handle from [dbo].[Manager_Config]")
        if not result:
            return None
        return result[0]

    def get_manager_config_dict(self):
        result = self.select_query_of_one_row("select Minutes_between_mails from [dbo].[Manager_Config]")
        if not result:
            return None
        self.set_manager_config_flag()
        config_dict = {"Minutes_between_mails": result[0]}
        return config_dict

    def set_manager_config_flag(self):
        self.update_query("update [dbo].[Manager_Config] set Handle = 0")

    def get_ip_port_config(self, table_name):
        result = self.select_query_of_one_row("select Manager_port, Manager_ip, Analayzer_port, Analayzer_ip, "
                                              "Camera_port, Camera_ip from [dbo].[Ip_port_components]")
        if not result:
            return None
        self.update_query("update [dbo].[Ip_port_components] set " + table_name + "_handle = 0")
        config_dict = {"Manager_port": result[0],
                       "Manager_ip": result[1],
                       "Analayzer_port": result[2],
                       "Analayzer_ip": result[3],
                       "Camera_port": result[4],
                       "Camera_ip": result[5]}
        return config_dict

    def set_ip_by_table_name(self, table_name):
        import socket
        my_ip = socket.gethostbyname(socket.gethostname())
        self.update_query("update [dbo].[Ip_port_components] set " + table_name + "_ip = '" + my_ip + "'")
        self.turn_on_components_ip_port_flags()

    def set_port_by_table_name(self, table_name, port):
        self.update_query("update [dbo].[Ip_port_components] set " + table_name + "_port = " + port)
        self.turn_on_components_ip_port_flags()

    def turn_on_components_ip_port_flags(self):
        self.update_query("update [dbo].[Ip_port_components] set Manager_handle = 1, "
                          "Analayzer_handle = 1, Camera_handle = 1")

    def get_flag_ip_port_by_table_name(self, table_name):
        result = self.select_query_of_one_row("select " + table_name + "_handle "
                                                                       "from [dbo].[Ip_port_components]")
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
