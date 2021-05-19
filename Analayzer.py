import time
import face_recognition
import os
import requests
from keras.models import load_model
from azure_sql_server import *
from detect_image import mask_image

NAME_COMPONENT = 'Analayzer'
b = Database()
b.set_ip_by_table_name(NAME_COMPONENT)


def update_config_ip_port(config):
    dict = b.get_ip_port_config(NAME_COMPONENT)
    for conf in dict:
        config[conf] = dict[conf]
    return config


def init_config():
    config = {}
    config["TIME_BETWEEN_SENDS"] = 30
    config = update_config_ip_port(config)
    print(config)
    return config


config = init_config()


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


def check_equal_images(known_image, unknown_image):
    path_known = save_image(known_image)
    path_unknown = save_image(unknown_image)
    try:
        known_image = face_recognition.load_image_file(path_known)
        unknown_image = face_recognition.load_image_file(path_unknown)
        biden_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
        print("results:      ", results)
        delete_image(path_known)
        delete_image(path_unknown)
    except:
        delete_image(path_known)
        delete_image(path_unknown)
        return False
    return results[0]


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


# model1 = load_model("models/model1.model")
# model2 = load_model("models/model2.model")
# print("load models")

#
# def mask_detect(image):
#     result1 = test_model_i(image, model1)
#     result2 = test_model_i(image, model2)
#     if (result1 == None and result2 == None):
#         return None
#     if (result1 == None):
#         result1 = 0
#     if (result2 == None):
#         result2 = 0
#     if ((result1 + result2) > 0):
#         return True
#     return False


def test_model_i(image, model):
    # model = load_model(path_to_model)
    labels_dict = {0: 'without mask', 1: 'mask'}
    color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
    size = 4
    # We load the xml file
    classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    image = cv2.flip(image, 1, 1)  # Flip to act as a mirror
    # Resize the image to speed up detection
    mini = cv2.resize(image, (image.shape[1] // size, image.shape[0] // size))
    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)
    label = -1
    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        face_img = image[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (150, 150))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(image, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(image, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return label


from flask import (
    Flask,
    render_template,
    jsonify, Response, request, json)

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


def convert_bytes_to_image(data):
    data = bytes(data.decode('utf8')[:-1], 'utf-8')
    image_64_decode = base64.decodebytes(data)
    image_result = open('testfile.jpg', 'wb')
    image_result.write(image_64_decode)
    image_result.close()
    image = cv2.imread('testfile.jpg')
    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)
    return image


def convert_image_to_varbinary(filename):
    image = open(filename, 'rb')
    image_read = image.read()
    image_64_encode = base64.encodebytes(image_read)
    image.close()
    return image_64_encode


def get_list_images(response):
    import requests
    result = response
    data = json.loads(result)
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


def check_config_ip_port():
    if b.get_flag_ip_port_by_table_name(NAME_COMPONENT) == '1':
        update_config_ip_port(config)
        print("after update")
        print(config)


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
            # print("before face")
            dict_id_workers_without_mask[id_worker] = face
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
            a = 1


def main():
    run_server()


main()
