from flask import Flask, render_template, flash, request,session
from cloudant.client import  Cloudant



client = Cloudant.iam("710709cf-9751-479d-8368-f7286e38749d-bluemix","h7TBqBQPUFRXQpxRRSchhmLXakubgKUOGL5jEJDTPYrs",connect=True)
my_database = client.create_database("database-dharan")


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'



@app.route("/")
def homepage():

    return render_template('index.html')



@app.route("/userhome")
def userhome():

    return render_template('userhome.html')
@app.route("/addamount")

@app.route("/NewUser")
def NewUser():

    return render_template('NewUser.html')







@app.route("/user")
def user():

    return render_template('user.html')


@app.route("/newuse",methods=['GET','POST'])
def newuse():
    if request.method == 'POST':#

        x = [x for x in request.form.values()]
        print(x)
        data = {
            '_id': x[1],
            'name': x[0],
            'psw': x[2]
        }
        print(data)
        query = {'_id': {'Seq': data['_id']}}
        docs = my_database.get_query_result(query)
        print(docs)
        print(len(docs.all()))
        if (len(docs.all()) == 0):
            url = my_database.create_document(data)
            return render_template('goback.html', data="Register, please login using your details")
        else:
            return render_template('goback.html', data="You are already a member, please login using your details")

@app.route("/userlog", methods=['GET', 'POST'])
def userlog():
        if request.method == 'POST':

            user = request.form['_id']
            passw = request.form['psw']
            print(user, passw)

            query = {'_id': {'$eq': user}}
            docs = my_database.get_query_result(query)
            print(docs)
            print(len(docs.all()))
            if (len(docs.all()) == 0):
                return render_template('goback.html', pred="The username is not found.")
            else:
                if ((user == docs[0][0]['_id'] and passw == docs[0][0]['psw'])):

                    return render_template("userhome.html")
                else:
                    return render_template('goback.html',data="user name and password incorrect")









@app.route("/vvideo", methods=['GET', 'POST'])
def vvideo():
    if request.method == 'POST':

        outttt = " No Drowing"
        ss ="person"


        file = request.files['fileupload']
        file.save('static/Out/Test.avi')

        import warnings
        warnings.filterwarnings('ignore')

        import time
        import cv2
        import os
        import numpy as np

        args = {"confidence": 0.5, "threshold": 0.3}
        flag = False

        labelsPath = "./yolo-coco/coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")
        final_classes = ['person']

        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                                   dtype="uint8")

        weightsPath = os.path.abspath("./yolo-coco/yolov3-tiny.weights")
        configPath = os.path.abspath("./yolo-coco/yolov3-tiny.cfg")

        # print(configPath, "\n", weightsPath)

        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        vs = cv2.VideoCapture('./static/Out/Test.avi')
        writer = None
        (W, H) = (None, None)

        flag = True

        flagg = 0

        while True:
            # read the next frame from the file
            (grabbed, frame) = vs.read()

            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break

            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > args["confidence"]:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                    args["threshold"])

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    if (ss=="person"):


                        flagg += 1
                        # print(flag)

                        if (flagg == 3):
                            outttt = "Drowing"
                            flagg = 0
                            import winsound

                            filename = 'alert.wav'
                            winsound.PlaySound(filename, winsound.SND_FILENAME)
                            sendmsg("9600713957","Drowing")

                            cv2.imwrite("alert.jpg", frame)



                        #color = [int(c) for c in COLORS[classIDs[0]]]
                        color ="red"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        text = "{}: {:.4f}".format("Person",
                                                   confidences[i])
                        cv2.putText(frame, text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            else:
                flag = True

            cv2.imshow("Output", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        # release the webcam and destroy all active windows
        vs.release()
        cv2.destroyAllWindows()

        return render_template('userhome.html', prediction=outttt)




def sendmsg(targetno,message):
    import requests
    requests.post("http://smsserver9.creativepoint.in/api.php?username=fantasy&password=596692&to=" + targetno + "&from=FSSMSS&message=Dear user  your msg is " + message + " Sent By FSMSG FSSMSS&PEID=1501563800000030506&templateid=1507162882948811640")













if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
