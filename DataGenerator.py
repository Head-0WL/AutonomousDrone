import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from pycocotools.coco import COCO
from skspatial.objects import Points
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


class DataGenerator():

    def __init__(self, coco_path, images_path, batch_size, input_size, cell_size, shuffle=False, augment=False):
        self.coco = COCO(coco_path)
        self.images_path = images_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.cell_size = cell_size
        self.stride = (cell_size[0] // 2, cell_size[1] // 2)
        self.shuffle = shuffle
        self.images = np.array(self.coco.dataset['images'])
        self.num_images = len(self.images)
        self.indexes = np.arange(self.num_images)
        self.augment = augment
        self.batch_index = 0

    def __call__(self):
        """
        :param labels_path: array paths to the labels
        :param batch_size: size of batch to use
        :return: yields one image with a corresponding label at a time
        """
        while True:
            # Select files (paths/indices) for the batch
            batch_indexes = self.indexes[self.batch_index:self.batch_index + self.batch_size]
            batch_images = self.images[batch_indexes]
            batch_input, batch_output = self.get_input_output(batch_images)

            self.batch_index += self.batch_size
            if self.batch_index > self.num_images - self.batch_size:
                self.batch_index = 0
                if self.shuffle:
                    np.random.shuffle(self.indexes)

            # Return a tuple of (input, output) to feed the network
            yield batch_input, batch_output

    def get_mask_cells(self, mask):
        shape = ((mask.shape[0] - self.cell_size[0]) + 1,
                 (mask.shape[1] - self.cell_size[1]) + 1,
                 self.cell_size[0],
                 self.cell_size[1])

        strides = 2 * mask.strides
        cells = np.lib.stride_tricks.as_strided(mask, shape=shape, strides=strides)

        return cells[::self.stride[0], ::self.stride[1]]

    def get_longest_line(self, cell_mask):
        """
        :param cell_mask: array of size cell_size
        :return: [length, coords] where length is the length of the longest
                line and coords are the coordinates of the endpoints
        """
        cell_mask = self.segment_to_line(cell_mask)
        points = cell_mask.nonzero()
        points = np.column_stack((points[1], points[0]))

        # Edge cases
        num_points = points.shape[0]
        if num_points == 0:
            return 0, [0, 0, 0, 0, 0]
        elif num_points == 1:
            return 0, np.hstack((1, points[0], points[0]))
        elif num_points == 2:
            dist = np.linalg.norm(points[0] - points[1])
            return dist, np.hstack((1, points[0], points[1]))

        # If points are collinear we cannot get the convex hull
        # Calculate distance between all pairs instead
        if Points(points).are_collinear():
            hdist = cdist(points, points, metric='euclidean')
            # Get the farthest apart points
            bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

            return hdist.max(), np.hstack((1, points[bestpair[0]], points[bestpair[1]]))

        # Find a convex hull in O(N log N)
        hull = ConvexHull(points)
        vertices = hull.vertices

        # Extract the points forming the hull
        hullpoints = points[vertices, :]

        # Find max distance between all pair of points on the hull
        # in O(H^2) time where H is number of points on hull
        hdist = cdist(hullpoints, hullpoints, metric='euclidean')

        # Get the farthest apart points
        bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

        return hdist.max(), np.hstack((1, hullpoints[bestpair[0]], hullpoints[bestpair[1]]))


    def segment_to_line(self, cell_mask):
        line_mask = np.zeros(cell_mask.shape, dtype=np.uint8)
        points = cell_mask.nonzero()
        points = np.column_stack((points[1], points[0]))

        if points.shape[0] > 0:
            [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * (vy / vx)) + y)
            righty = int(((cell_mask.shape[1] - x) * (vy / vx)) + y)
            cv2.line(line_mask, (cell_mask.shape[1] - 1, righty), (0, lefty), [255, 255, 255], 1)

        return cv2.bitwise_and(line_mask, line_mask, mask=cell_mask)


    def get_batch_regressor_labels(self, batch_images):
        grid_height = (self.input_size[0] - self.cell_size[0]) // self.stride[1] + 1
        grid_width = (self.input_size[1] - self.cell_size[1]) // self.stride[0] + 1

        batch_endpoints = []
        for b in range(self.batch_size):
            segmentations = self.coco.imgToAnns[batch_images[b]['id']]
            longest = np.zeros((grid_height, grid_width))
            endpoints = np.zeros((grid_height, grid_width, 5))

            for segmentation in segmentations:
                mask = self.coco.annToMask(segmentation)
                cells = self.get_mask_cells(mask)

                for i in range(cells.shape[0]):
                    for j in range(cells.shape[1]):
                        length, coords = self.get_longest_line(cells[i, j])

                        if length > longest[i, j]:
                            longest[i, j] = length
                            endpoints[i, j] = coords

            batch_endpoints.append(endpoints)

        return np.array(batch_endpoints)

    def random_augment(self, img):
        if tf.random.uniform([]) < 0.5:
            tfa.image.gaussian_filter2d(img, sigma=2.0)

        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.8, 1.25)
        img = tf.image.random_saturation(img, 0.8, 1.25)
        img = tf.image.random_hue(img, 0.02)

        return img


    def get_input_output(self, batch_images):
        images = []
        for image in batch_images:
            # Read input image
            # print(image['file_name'])
            img = cv2.imread(self.images_path + image['file_name'])
            if self.augment:
                img = self.random_augment(img)
            images.append(img)

        regressor_labels = self.get_batch_regressor_labels(batch_images)
        classifier_labels = np.concatenate((np.expand_dims(regressor_labels[:, :, :, 0], axis=3),
                                            1 - np.expand_dims(regressor_labels[:, :, :, 0], axis=3)), axis=3)

        return tf.convert_to_tensor(images), {'classifier': classifier_labels.astype('float32'),
                                              'regressor': regressor_labels}
