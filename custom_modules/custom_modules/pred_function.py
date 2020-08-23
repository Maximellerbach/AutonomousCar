import cv2
import numpy as np
from tqdm import tqdm

from .visutils import plot

# TODO refactor those functions broke it by changing dataset


def compare_pred(self, Dataset, dos, dt_range=(0, -1)):
    paths = Dataset.generator_load_dataset_sorted(dos)
    paths = paths[dt_range[0]:dt_range[1]]

    preds = []
    annotations = []
    for path in tqdm(paths):
        img, annotation = Dataset.load_img_and_annotation(path)
        annotations.append(annotation)

        inputs = [np.expand_dims(img/255, axis=0)]
        for index in self.input_components:
            inputs.append(np.expand_dims(annotations[index], axis=0))

        pred = self.model.predict(inputs)[0]
        preds.append(pred)

    preds = np.array(preds)
    annotations = np.array(annotations)

    for index in self.output_components:
        plot.plot_time_series(annotations[:, index], preds[:, index])


def pred_img(self, Dataset, path, sleeptime):
    img, annotations = Dataset.load_img_and_annotation(path)
    to_pred_img = np.expand_dims(img/255, axis=0)

    inputs = [to_pred_img]
    for index in self.input_components:
        inputs.append(np.expand_dims())

    pred = self.model.predict(inputs)[0]

    '''
    c = np.copy(img)
    cv2.line(c,
             (img.shape[1]//2, img.shape[0]),
             (int(img.shape[1] / 2+average*30), img.shape[0]-50),
             color=[255, 0, 0], thickness=4)/255

    cv2.imshow('img', c)
    cv2.waitKey(sleeptime)
    '''


def load_frames(self, path, size=(160, 120), batch_len=32):
    """
    load a batch of frame from video
    """
    batch = []

    for _ in range(batch_len):
        _, frame = self.cap.read()
        frame = cv2.resize(frame, size)
        batch.append(frame)

    return batch


def after_training_test_pred(self, dos='C:\\Users\\maxim\\random_data\\4 trackmania A04\\', size=(160, 120), nimg_size=(5, 5), sleeptime=1):
    Dataset = dataset.Dataset(
        [dataset.direction_component, dataset.speed_component, dataset.time_component])
    for i in Dataset.load_dos(dos):
        img = cv2.imread(i)
        if self.load_speed:
            speed = Dataset.load_component_item(i, 1)
            pred_img(self, img, size, sleeptime,
                     speed=speed, nimg_size=nimg_size)


def speed_impact(self, dos, dt_range=(0, -1), sleeptime=33):

    Dataset = dataset.Dataset(
        [dataset.direction_component, dataset.speed_component, dataset.time_component])
    paths = Dataset.load_dos_sorted(dos, sort_component=-1)
    paths = paths[dt_range[0]:dt_range[1]]
    dts_len = len(paths)

    Y = np.array(Dataset.repeat_function(Dataset.load_annotation, paths))[:, 0]
    for it, path in enumerate(paths):
        img = cv2.imread(path)/255
        img_pred = np.expand_dims(img, axis=0)

        original_speed = Dataset.load_component_item(path, 1)
        original_pred = self.model.predict(
            [img_pred, np.expand_dims(original_speed, axis=0)])

        if self.load_speed[1]:
            original_pred, throttle_pred = original_pred
            original_pred = original_pred[0]
            throttle_pred = throttle_pred[0]
        real_lab = Y[it]

        c = img.copy()
        if self.load_speed[1]:
            cv2.line(c, (img.shape[1]//2, img.shape[0]),
                     (int(img.shape[1]/2+original_pred*30),
                      img.shape[0]-int(throttle_pred*50)),
                     color=[1, 0, 0], thickness=3)
        else:
            cv2.line(c, (img.shape[1]//2, img.shape[0]),
                     (int(img.shape[1]/2 + original_pred*30), img.shape[0]-50),
                     color=[0, 1, 1], thickness=3)

        cv2.line(c,
                 (img.shape[1]//2, img.shape[0]),
                 (int(img.shape[1] / 2 + real_lab*30), img.shape[0]-50),
                 color=[0, 0, 1], thickness=2)

        modified = compute_speed(self.model, img_pred,
                                 real_lab, accuracy=0.5, values_range=(0, 21))

        for it, pred in enumerate(modified):
            angle = pred[0][0][0]
            modified_throttle = pred[1][0][0]
            cv2.line(c, (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+angle*30), img.shape[0]-int(
                modified_throttle*50)), color=[0.5+(it)/(2*len(modified)), (it)/len(modified), 0], thickness=1)

        cv2.imshow('img', img)
        cv2.imshow('angles', c)
        cv2.waitKey(sleeptime)
    cv2.destroyAllWindows()


def compute_speed(model, img_pred, original_pred, accuracy=1, values_range=(0, 20), show=True):
    speeds = np.array(range(values_range[0], int(
        values_range[1]*accuracy)))/accuracy
    modified = []
    for speed in speeds:
        modified.append(model.predict(
            [img_pred, np.expand_dims(speed, axis=0)]))

    return modified


def process_trajectory_error():  # TODO: evaluate long term precision of the model
    return
