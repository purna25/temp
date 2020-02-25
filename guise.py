import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import cv2
from PIL import Image
import argparse
from imutils.video import VideoStream


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class LicensePlateDetection:
    def __init__(self, model_path):
        self.__model = load_model(model_path)
        self.__input_w = 416
        self.__input_h = 416
        self.__anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        self.__class_threshold = 0.6
        self.__labels = ["auto", "bike", "bus", "car", "cycle", "dog", "jeep", "lorry", "person", "rider", "tractor"]
        # ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        #     "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        #     "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        #     "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        #     "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        #     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        #     "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        #     "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        #     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        #     "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        # # ["LP"]
        # labels = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",
        # "Q","R","S","T","U","V","W","X","Y","Z"]
        self.__matrix = np.zeros((416, 416, 3))


    def __sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def __decode_netout(self, netout, anchors):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2] = self.__sigmoid(netout[..., :2])
        netout[..., 4:] = self.__sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > self.__class_threshold

        for i in range(grid_h * grid_w):
            row = i / grid_w
            col = i % grid_w
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                if objectness.all() <= self.__class_threshold: continue
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w  # center position, unit: image width
                y = (row + y - 0.85) / grid_h  # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / self.__input_w  # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / self.__input_h  # unit: image height
                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]
                box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                boxes.append(box)
        return boxes

    def __correct_yolo_boxes(self, boxes, image_h, image_w):
        new_w, new_h = self.__input_w, self.__input_h
        for i in range(len(boxes)):
            x_offset, x_scale = (self.__input_w - new_w) / 2. / self.__input_w, float(new_w) / self.__input_w
            y_offset, y_scale = (self.__input_h - new_h) / 2. / self.__input_h, float(new_h) / self.__input_h
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    def __interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def __bbox_iou(self, box1, box2):
        intersect_w = self.__interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self.__interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
        union = w1 * h1 + w2 * h2 - intersect
        return float(intersect) / union

    def __do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i].classes[c] == 0: continue
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if self.__bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0

    # load and prepare an image
    def __load_image_pixels(self, frame):
        # resize and convert to numpy array
        image = cv2.resize(frame, (self.__input_w, self.__input_h), interpolation=cv2.INTER_LINEAR_EXACT)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_array(image)
        # scale pixel values to [0, 1]
        image = image.astype('float32')
        image /= 255.0
        # add a dimension so that we have one sample
        image = expand_dims(image, 0)
        return image

    # get all of the results above a threshold
    def __get_boxes(self, boxes):
        v_boxes, v_labels, v_scores = list(), list(), list()
        # enumerate all boxes
        for box in boxes:
            # enumerate all possible labels
            for i in range(len(self.__labels)):
                # check if the threshold for this label is high enough
                if box.classes[i] > self.__class_threshold:
                    v_boxes.append(box)
                    v_labels.append(self.__labels[i])
                    v_scores.append(box.classes[i] * 100)
                # don't break, many labels may trigger for one box
        return v_boxes, v_labels, v_scores

    def __draw_box(self, image, v_boxes, v_labels, v_scores):
        for i in range(len(v_boxes)):
            box = v_boxes[i]
            # get coordinates
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            # calculate width and height of the box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # width, height = x2 - x1, y2 - y1
            # create the shape
            # rect = Rectangle((x1, y1), width, height, fill=False, color='white')
            # draw the box
            # ax.add_patch(rect)
            # draw text and score in top left corner
            label = "%s (%.3f)" % (v_labels[i], v_scores[i])
            cv2.putText(image, label,(x1-15,y1-15),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            # pyplot.text(x1, y1, label, color='white')
        return image
    

    def run(self, frame):
        image = self.__load_image_pixels(frame)
        yhat = self.__model.predict(image)
        # print([a.shape for a in yhat])
        boxes = list()

        for i in range(len(yhat)):
            # decode the output of the network
            boxes += self.__decode_netout(yhat[i][0], self.__anchors[i])
        # correct the sizes of the bounding boxes for the shape of the image
        self.__correct_yolo_boxes(boxes, frame.shape[0], frame.shape[1])
        # suppress non-maximal boxes
        self.__do_nms(boxes, 0.5)
        v_boxes, v_labels, v_scores = self.__get_boxes(boxes)
        print(v_labels)
        image = self.__draw_box(frame, v_boxes, v_labels, v_scores)
        return image

        # summarize what we found
        # for i in range(len(v_boxes)):
        #     print(v_labels[i], v_scores[i])

ap = argparse.ArgumentParser()
ap.add_argument("--cam", "-c", type=int, help="camera id for webcam", default=0)
ap.add_argument("--video", "-v", type=str, help="path for a video")

args = ap.parse_args()

print(args)

if args.cam:
    vid = VideoStream(args.cam)
elif args.video:
    vid = cv2.VideoCapture(args.video)
else:
    vid = cv2.VideoCapture(0)

# # if not args.cam is None:
# vid = cv2.VideoCapture(args.cam)


win = cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

while True:
    ret, frame = vid.read()
    if ret:
        model = LicensePlateDetection("guisenet_obj.h5")
        image = model.run(frame)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            vid.release()
            break



