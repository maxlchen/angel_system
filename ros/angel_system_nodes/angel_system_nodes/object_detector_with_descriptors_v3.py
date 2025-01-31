from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from threading import Lock
import torch
import torchvision

print("TORCH VERSION", torch.__version__)
print("torch")
from torch.autograd import Variable
from torchvision.ops import nms

from angel_msgs.msg import ObjectDetection2dSet
from angel_system.fasterrcnn.faster_rcnn.resnet import resnet
from angel_system.fasterrcnn.processing_utils import _get_image_blob
from angel_system.fasterrcnn.rpn.bbox_transform import clip_boxes
from angel_system.fasterrcnn.rpn.bbox_transform import bbox_transform_inv
from angel_utils.conversion import time_to_int
import cv2


BRIDGE = CvBridge()


class ObjectDetectorWithDescriptors_v3(Node):
    """
    ROS node that subscribes to `Image` messages and publishes detections and
    descriptors for those images in the form of `ObjectDetectionSet2d` messages.

    This version is equivalent to v2 except for an attempt to optimize the
    post-processing method to remove python for-loops and make use of tensor
    operations instead. This has tested to be ever so slightly faster than v2,
    but not in a way that addresses the primary bottlenecks.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.count = 0
        self._image_topic = (
            self.declare_parameter("image_topic", "PVFramesRGB")
            .get_parameter_value()
            .string_value
        )
        self._desc_topic = (
            self.declare_parameter("descriptor_topic", "ObjectDetections")
            .get_parameter_value()
            .string_value
        )
        self._min_time_topic = (
            self.declare_parameter("min_time_topic", "ObjDetMinTime")
            .get_parameter_value()
            .string_value
        )
        self._torch_device = (
            self.declare_parameter("torch_device", "cuda")
            .get_parameter_value()
            .string_value
        )
        self._detection_threshold = (
            self.declare_parameter("detection_threshold", 0.5)
            .get_parameter_value()
            .double_value
        )
        self._object_vocabulary = (
            self.declare_parameter(
                "object_vocab_list",
                "/angel_workspace/model_files/fasterrcnn_fine_tuned_label_list.txt",
            )
            .get_parameter_value()
            .string_value
        )
        self._model_checkpoint = (
            self.declare_parameter(
                "model_checkpoint",
                "/angel_workspace/model_files/faster_rcnn_res101_vg.pth",
            )
            .get_parameter_value()
            .string_value
        )

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Detection threshold: {self._detection_threshold}")
        log.info(f"Torch device? {self._torch_device}")
        self.first = True
        self._model = None
        self._model_device = None

        self._subscription = self.create_subscription(
            Image, self._image_topic, self.image_callback, 1
        )
        self._publisher = self.create_publisher(
            ObjectDetection2dSet, self._desc_topic, 1
        )

        # Be able to receive a notification that there is a minimum time
        # before which we should not detect objects.
        self._min_time_subscription = self.create_subscription(
            Time, self._min_time_topic, self.receive_min_time, 1
        )
        # minimum time in nanoseconds (see `time_to_int`)
        self._min_time_lock = Lock()
        self._min_time: int = 0

        # Load class labels
        classes = ["__background__"]
        with open(self._object_vocabulary) as f:
            for obj in f.readlines():
                classes.append(obj.split(",")[0].lower().strip())
        self.classes = classes
        self.counter = 0
        log.info("Ready to detect")

    def get_model(self) -> torch.nn.Module:
        """
        Lazy load the torch model in an idempotent manner.
        :raises RuntimeError: Use of CUDA was requested but is not available.
        """
        model = self._model
        if model is None:
            print("classes", self.classes)
            model = resnet(self.classes, pretrained=True, class_agnostic=False)
            model.create_architecture()
            print("checkpoint", self._model_checkpoint)
            checkpoint = torch.load(self._model_checkpoint)
            model.load_state_dict(checkpoint["model"])
            model.eval()

            # Transfer the model to the requested device
            model_device = torch.device(device=self._torch_device)
            model = model.to(device=model_device)

            self._model = model
            self._model_device = model_device

        return model

    def receive_min_time(self, msg: Time) -> None:
        """
        Set a minimum time for frame processing.

        If the given time is older than an already set minimum time, ignore
        this message.
        """
        with self._min_time_lock:
            msg_ns = time_to_int(msg)
            self.get_logger().info(f"Received new min frame time: {msg_ns} ns")
            self._min_time = max(self._min_time, msg_ns)

    def get_min_time(self) -> int:
        """
        Get the minimum time (in nanoseconds) that new images must be more
        recent than to be considered for processing.

        We should only process frames whose associated timestamp is greater
        than this (in nanoseconds).
        """
        with self._min_time_lock:
            return self._min_time

    def image_callback(self, image):
        """
        Callback function for the image subscriber. Performs image preprocessing,
        model inference, output postprocessing, and detection set publishing for
        each image received.

        This callback may return early if we receive an image that is before
        a received min processing time.
        """
        log = self.get_logger()
        model = self.get_model()
        img_time_ns = time_to_int(image.header.stamp)
        min_time = self.get_min_time()
        # if img_time_ns <= min_time:
        # Before min processing time, don't process this frame.
        # log.warn(f"Skipping frame with time {img_time_ns} ns <= min time "
        #         f"{min_time} ns")
        # return

        # log.info(f"Starting detection for frame time {img_time_ns} ns")

        # Preprocess image - NOTE: bgr order required by _get_image_blob
        im_in = np.array(BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8"))
        im_data, im_info, gt_boxes, num_boxes, im_scales = self.preprocess_image(im_in)

        # Send to model
        with torch.no_grad():
            (rois, cls_prob, bbox_pred, _, _, _, _, _, pooled_feat) = model(
                im_data, im_info, gt_boxes, num_boxes, pool_feat=True
            )

        # Postprocess model output
        detection_info = self.postprocess_detections(
            rois, cls_prob, bbox_pred, im_info, pooled_feat, im_scales
        )

        # The above may take non-trivial time.
        # Check if we were given a min frame processing time that exceeds the
        # stamp of the frame just processed.
        if img_time_ns < self.get_min_time():
            # min processing time is now beyond us, don't output for this frame.
            return

        # Form into object detection set message
        msg = ObjectDetection2dSet()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = image.header.frame_id
        msg.source_stamp = image.header.stamp

        if detection_info["boxes"] is None:
            msg.num_detections = 0
        else:
            msg.num_detections = len(detection_info["labels"])
            msg.label_vec = self.classes[1:]
            msg.label_confidences = detection_info["scores"].ravel().tolist()

            msg.left = detection_info["boxes"][:, 0].tolist()
            msg.top = detection_info["boxes"][:, 1].tolist()
            msg.right = detection_info["boxes"][:, 2].tolist()
            msg.bottom = detection_info["boxes"][:, 3].tolist()

            msg.descriptor_dim = detection_info["feats"].shape[-1]
            msg.descriptors = detection_info["feats"].ravel().tolist()

        # Publish detection set message
        self._publisher.publish(msg)
        if self.counter >= 30:
            self.counter = 0
            log.info("published 30 messages")
        else:
            self.counter += 1
        # log.info("Published detection set message")
        self.first = False

    def preprocess_image(self, im_in):
        """
        Preprocess the image and return the necessary inputs for the fasterrcnn
        mode. Returns the image blob data, image info, ground truth boxes, image
        scales, and number of boxes.

        Based on:
        https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome/blob/master/generate_tsv.py
        """
        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1).to(device=self._torch_device)
        im_info = torch.FloatTensor(1).to(device=self._torch_device)
        num_boxes = torch.LongTensor(1).to(device=self._torch_device)
        gt_boxes = torch.FloatTensor(1).to(device=self._torch_device)

        # Make variable
        with torch.no_grad():
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

        blobs, im_scales = _get_image_blob(im_in)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
        )

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()

        return im_data, im_info, gt_boxes, num_boxes, im_scales

    def postprocess_detections(
        self, rois, cls_prob, bbox_pred, im_info, pooled_feat, im_scales
    ):
        """
        Form the model outputs into a dictionary containing the bounding boxes,
        object indices, labels, features, and probability scores.
        Based on:
        https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome/blob/master/generate_tsv.py
        """
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
            (0.1, 0.1, 0.2, 0.2)
        ).to(device=self._torch_device) + torch.FloatTensor((0.0, 0.0, 0.0, 0.0)).to(
            device=self._torch_device
        )
        box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        if self.first:
            print(pred_boxes.shape)
        pred_boxes /= im_scales[0]
        # Simply repeat the boxes, once for each class
        scores = scores.squeeze()
        # -------------
        pred_boxes = pred_boxes.squeeze()

        const_bbox_size = torch.arange(4).to(device=self._torch_device)
        max_class_ind = (torch.argmax(scores[:, 1:], dim=1) + 1).to(
            device=self._torch_device
        )
        # getting indices for bbox in flattened array length num_classes*4
        bbox_indices = (max_class_ind.unsqueeze(1) * 4) + const_bbox_size
        max_class_boxes = pred_boxes.gather(1, bbox_indices)
        max_class_scores = scores.gather(1, max_class_ind.unsqueeze(1)).squeeze()
        above_thresh_mask = max_class_scores >= self._detection_threshold
        above_thresh_max_class_conf = max_class_scores[above_thresh_mask]
        above_thresh_scores = scores[above_thresh_mask]
        above_thresh_boxes = max_class_boxes[above_thresh_mask]
        above_thresh_class_ind = max_class_ind[above_thresh_mask]
        above_thresh_feat = pooled_feat[above_thresh_mask]
        keep = nms(above_thresh_boxes, above_thresh_max_class_conf, 0.3)
        # downsample scores and boxes based on nms

        if keep.numel():
            keep_boxes = above_thresh_boxes[keep]
            keep_class_ind = above_thresh_class_ind[keep]
            keep_scores = above_thresh_scores[keep]
            keep_feat = above_thresh_feat[keep]

            objects = keep_class_ind - 1
            box_dets = keep_boxes.cpu().numpy()
            scores = keep_scores[:, 1:]
            labels = []
            classes = self.classes
            for i in keep_class_ind:
                labels.append(classes[i])
            feats = keep_feat.cpu().detach().numpy()

            sample_info = dict(
                labels=labels,
                boxes=box_dets.astype("float32"),
                objects=objects.cpu().numpy(),
                feats=feats,
                scores=scores,
            )
        else:
            sample_info = dict(
                boxes=None, objects=None, feats=None, labels=None, scores=None
            )
        return sample_info
        # ------------------
        max_conf = torch.zeros((scores.shape[0])).to(device=self._torch_device)
        inds = torch.nonzero(scores > self._detection_threshold, as_tuple=True)
        cls_scores = scores[inds]
        _, order = torch.sort(cls_scores, 0, True)

        filled_boxes = boxes[:, inds[0], :]
        if filled_boxes.ndim == 1:
            filled_boxes = torch.unsqueeze(filled_boxes, 0)
        elif filled_boxes.ndim > 2:
            filled_boxes = torch.squeeze(filled_boxes, 0)
        keep = nms(filled_boxes[order, :], cls_scores[order], 0.3)

        index = inds[0][order[keep]]
        max_conf = torch.where(
            torch.max(scores[index, :]) > max_conf[index],
            torch.max(scores[index, :]),
            max_conf[index],
        )
        keep_boxes = torch.where(
            max_conf >= self._detection_threshold,
            max_conf,
            torch.tensor(0.0).to(device=self._torch_device),
        )

        keep_boxes = torch.squeeze(torch.nonzero(keep_boxes))
        if keep_boxes.numel():
            if keep_boxes.ndim == 0:
                objects = torch.argmax(scores[keep_boxes][1:])
                objects = torch.unsqueeze(objects, 0)
                box_dets = np.zeros((1, 4))
                boxes = filled_boxes[keep_boxes]
                kind = objects + 1

                bbox = boxes[kind * 4 : (kind + 1) * 4]
                box_dets[0] = bbox

                scores = scores[keep_boxes][1:]
                scores = torch.unsqueeze(scores, 0)
                labels = []
                for i in objects:
                    labels.append(self.classes[i + 1])

                feats = pooled_feat[keep_boxes].cpu().detach().numpy()
                feats = np.expand_dims(feats, 0)
            else:
                objects = torch.argmax(scores[keep_boxes][:, 1:], dim=1)
                box_dets = np.zeros((len(keep_boxes), 4))
                box_dets = filled_boxes[keep_boxes].cpu().numpy()

                scores = scores[keep_boxes][:, 1:]
                labels = []
                for i in objects:
                    labels.append(self.classes[i + 1])

                feats = pooled_feat[keep_boxes].cpu().detach().numpy()

            sample_info = dict(
                labels=labels,
                boxes=box_dets.astype("float32"),
                objects=objects.cpu().numpy(),
                feats=feats,
                scores=scores,
            )
        else:
            sample_info = dict(
                boxes=None, objects=None, feats=None, labels=None, scores=None
            )

        return sample_info


def main():
    rclpy.init()

    node = ObjectDetectorWithDescriptors_v3()
    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
