import time
import face_recognition
import os
import requests
from keras.models import load_model
from azure_sql_server import *
from detect_image import mask_image
from requests import get
import socket
from flask import (
    Flask, json)

NAME_COMPONENT = 'Analayzer'
PORT_COMPONENT = '5002'
b = Database()
b.set_ip_by_table_name(NAME_COMPONENT)
b.set_port_by_table_name(NAME_COMPONENT, PORT_COMPONENT)

print("ip: ", socket.gethostbyname(socket.gethostname()))

ip = get('https://api.ipify.org').text
print('My public IP address is: {}'.format(ip))


# After the change of the flag, update the config.
def update_config_ip_port(config):
    dict = b.get_ip_port_config(NAME_COMPONENT)
    for conf in dict:
        config[conf] = dict[conf]
    return config


# Init the config at the first time running.
def init_config():
    config = {}
    config["TIME_BETWEEN_SENDS"] = 30
    config = update_config_ip_port(config)
    print(config)
    return config


config = init_config()


# Get dictionary of the ids workers with their photos.
def get_dictionary_workers():
    dict = {}
    try:
        dict = b.get_workers_to_images_dict()
    except:
        print("can't get dictionary workers")
    return dict


# Get image, save local, return path.
def save_image(img):
    if not os.path.exists('saved_pictures'):
        os.makedirs('saved_pictures')
    import time
    path_to_save = "saved_pictures/face%s.jpg" % str(time.time())
    cv2.imwrite(path_to_save, img)
    return path_to_save


# Get path to image and delete.
def delete_image(path_to_image):
    try:
        os.remove(path_to_image)
    except:
        print("The image doesn't exist")


# Get 2 images and check if this is the same person. Return true or false.
def check_equal_images(known_image, unknown_image):
    path_known = save_image(known_image)
    path_unknown = save_image(unknown_image)
    try:
        known_image = face_recognition.load_image_file(path_known)
        unknown_image = face_recognition.load_image_file(path_unknown)
        known_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        print("results:      ", results)
        delete_image(path_known)
        delete_image(path_unknown)
    except:
        delete_image(path_known)
        delete_image(path_unknown)
        return False
    return results[0]


# Get image of person without mask and dictionary of the workers. Return the id of this person.
# If not found, return -1.
def get_id_worker(face, dict_workers):
    # dict_workers = get_dictionary_workers()
    for key in dict_workers:
        if check_equal_images(dict_workers[key], face):
            return key
    return -1


# Get user supplied values
casc_path = "haarcascade_frontalface_default.xml"
# Create the haar cascade
face_cascade = cv2.CascadeClassifier(casc_path)


# Get image, return list of faces that in this image.
def get_list_faces(image):
    # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    print("Found {0} faces!".format(len(faces)))
    list_images = []
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        im1 = image[y:y + h, x:x + w]
        list_images.append(im1)
    return list_images


# Create the application instance
app = Flask(__name__, template_folder="templates")


@app.route('/', methods=['POST'])
def result():
    print("post")
    data = request.values
    for key in data:
        dict = data[key]
    try:
        list_images = get_list_images(dict)
    except:
        list_images = []
    try:
        analayzer(list_images)
    except:
        print("Problem with analyzer")
    return "OK"


# Get bytes of image, and convert it to image.
def convert_bytes_to_image(data):
    data = bytes(data.decode('utf8')[:-1], 'utf-8')
    image_64_decode = base64.decodebytes(data)
    image_result = open('testfile.jpg', 'wb')
    image_result.write(image_64_decode)
    image_result.close()
    image = cv2.imread('testfile.jpg')
    return image


# Get path to image and convert it to varbinary for sending. Return the converted image.
def convert_image_to_varbinary(filename):
    image = open(filename, 'rb')
    image_read = image.read()
    image_64_encode = base64.encodebytes(image_read)
    image.close()
    return image_64_encode


def get_list_images(response):
    data = json.loads(response)
    list_images = []
    for key in data:
        decoded_image_data = base64.decodebytes(bytes(data[key], encoding='utf8'))
        list_images.append(convert_bytes_to_image(decoded_image_data))
    return list_images


def convert_dict_for_sending(dict):
    for key in dict:
        path = save_image(dict[key])
        dict[key] = convert_image_to_varbinary(path)
        dict[key] = base64.encodebytes(dict[key]).decode('utf-8')
    return dict


def get_url_manager():
    url = 'http://' + config['Manager_ip'] + ':' + config['Manager_port'] + '/'
    return url


def post_ids_to_manager(dict={}):
    dict = convert_dict_for_sending(dict)
    dict = json.dumps(dict)
    url = 'http://127.0.0.1:5004/'
    url = get_url_manager()
    x = requests.post(url, data={'dict': dict})
    print(x)


dict_workers = {}
is_init_dict_workers = False


# Check if the flag of the config was chnaged and update if the flag equal to 1.
def check_config_ip_port():
    if b.get_flag_ip_port_by_table_name(NAME_COMPONENT) == '1':
        update_config_ip_port(config)
        print("after update")
        print(config)


# def check_if_dictionary_workers_changed():
# if b.get_flag_dictionary_workers_changed():
#     return get_dictionary_workers()
# return dict_workers


def analayzer(list_images):
    check_config_ip_port()
    print("analayzer")
    time_before = time.time()
    dict_id_workers_without_mask = {}
    for image in list_images:
        list_faces = get_list_faces(image)
        for face in list_faces:
            try:
                res = mask_image(face)
            except:
                print("problem with result model")
                res = "No Mask"
            print("result: ", res)
            if res == "Mask":
                print("with mask")
                continue
            print("without mask")
            global is_init_dict_workers
            global dict_workers
            if not is_init_dict_workers:
                print("get dictionary workers")
                dict_workers = get_dictionary_workers()
                if len(dict_workers) > 0:
                    is_init_dict_workers = True
            # If there is no match, return -1.
            id_worker = get_id_worker(face, dict_workers)
            print("id: ", id_worker)

            if id_worker == -1:
                continue
            dict_id_workers_without_mask[id_worker] = image
            # print("after face")
    is_init_dict_workers = False
    import copy
    post_ids_to_manager(copy.deepcopy(dict_id_workers_without_mask))
    print("time after from db: ", time.time() - time_before)


from flask import Flask, jsonify, request
import json, os, signal


@app.route('/stop_server', methods=['GET'])
def stopServer():
    print("stopppp")
    os.kill(os.getpid(), signal.SIGINT)
    print("get pid")
    return jsonify({"success": True, "message": "Server is shutting down..."})


def run_server():
    print("hi analayzer")
    while True:
        try:
            from waitress import serve
            serve(app, host=config[NAME_COMPONENT + '_ip'], port=int(config[NAME_COMPONENT + '_port']))
            # app.run(port=5002, debug=True)
        except:
            print("There is a problem with starting the analyzer")


def main():
    run_server()


if __name__ == '__main__':
    main()
