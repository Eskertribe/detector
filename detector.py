import cv2
import imutils
import numpy as np
import requests
import argparse

url = "http://192.168.178.20:8080/shot.jpg"


def detect(frame):
    bodies = upperBodyCascade.detectMultiScale(
        frame, 1.3, 5)

    person = 1
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    cv2.putText(frame, 'Status : Detecting ', (40, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}',
                (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)

    return frame


def detectByCamera(writer):
    video = cv2.VideoCapture("http://192.168.178.20:8080/shot.jpg")

    while True:
        check, frame = video.read()

        frame = detect(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def detectByPathImage(path, output_path):
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)
        cv2.imshow("Android_cam", img)

        # Press Esc key to exit
        if cv2.waitKey(1) == 27:
            print("--- BREAK ---")
            cv2.destroyAllWindows()
            break
        # image = cv2.imread(path)

        # image = imutils.resize(image, width = min(800, image.shape[1]))

        result_image = detect(img)

        if output_path is not None:
            cv2.imwrite(output_path, result_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def humanDetector(args):
    image_path = args["image"]
    video_path = args['video']
    if str(args["camera"]) == 'true':
        camera = True
    else:
        camera = False

    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(
            args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(ouput_path, writer)
    elif image_path is not 1:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])


def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None,
                           help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None,
                           help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=False,
                           help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str,
                           help="path to optional output video file")
    args = vars(arg_parse.parse_args())

    return args


if __name__ == "__main__":
    upperBodyCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_upperbody.xml ")
    # HOGCV = cv2.HOGDescriptor()
    # HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = argsParser()
    humanDetector(args)
