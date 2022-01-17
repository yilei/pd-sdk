import concurrent
import hashlib
import logging
import uuid
from collections import defaultdict
from concurrent.futures import Future
from datetime import datetime
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pypeln

from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP_INV, DATETIME_FORMAT, POINT_FORMAT, DirectoryName
from paralleldomain.common.dgp.v0.dtos import (
    AnnotationsBoundingBox2DDTO,
    AnnotationsBoundingBox3DDTO,
    BoundingBox2DDTO,
    BoundingBox3DDTO,
    CalibrationDTO,
    CalibrationExtrinsicDTO,
    CalibrationIntrinsicDTO,
    OntologyFileDTO,
    PoseDTO,
    RotationDTO,
    SceneDataDatumImage,
    SceneDataDatumPointCloud,
    SceneDataDatumTypeImage,
    SceneDataDatumTypePointCloud,
    SceneDataDTO,
    SceneDataIdDTO,
    SceneDTO,
    SceneSampleDTO,
    SceneSampleIdDTO,
    TranslationDTO,
)
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.encoding.dgp.transformer import (
    BoundingBox2DTransformer,
    BoundingBox3DTransformer,
    InstanceSegmentation2DTransformer,
    InstanceSegmentation3DTransformer,
    OpticalFlowTransformer,
    SemanticSegmentation2DTransformer,
    SemanticSegmentation3DTransformer,
)
from paralleldomain.encoding.encoder import ENCODING_THREAD_POOL, SceneEncoder
from paralleldomain.encoding.encoder_v2 import (
    DatasetEncoderV2,
    PartialEncoder,
    PartialEncoderName,
    PartialsSummarizer,
    T,
)
from paralleldomain.model.annotation import Annotation, AnnotationType, AnnotationTypes, BoundingBox2D, BoundingBox3D
from paralleldomain.model.dataset import Dataset
from paralleldomain.model.sensor import CameraModel, CameraSensorFrame, LidarSensorFrame, SensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import write_json


class DGPCameraEncoder(PartialEncoder[SensorFrame[datetime]]):
    def __init__(
        self,
        camera_names: Optional[List[SensorName]] = None,
        frame_ids: Optional[List[FrameId]] = None,
        save_as_camera_name: Optional[Dict[SensorName, SensorName]] = None,
    ):
        self.save_as_camera_name = save_as_camera_name
        self.frame_ids = frame_ids
        self.camera_names = camera_names

    def encode(self, item: SensorFrame[datetime]) -> Optional[Tuple[SensorName, Dict[str, SceneDataDTO]]]:
        if not isinstance(item, CameraSensorFrame):
            return
        elif self.frame_ids is not None and item.frame_id not in self.frame_ids:
            return
        elif self.camera_names is not None and item.sensor_name not in self.camera_names:
            return

    def _encode_camera_frame(
        self, frame_id: str, camera_frame: CameraSensorFrame[datetime], last_frame: Optional[bool] = False
    ) -> Tuple[str, Dict[str, Dict[str, Future]]]:
        return frame_id, dict(
            annotations={
                "0": self._encode_bounding_boxes_2d(sensor_frame=camera_frame)
                if AnnotationTypes.BoundingBoxes2D in camera_frame.available_annotation_types
                and AnnotationTypes.BoundingBoxes2D in self._annotation_types
                else None,
                "1": self._encode_bounding_boxes_3d(sensor_frame=camera_frame)
                if AnnotationTypes.BoundingBoxes3D in camera_frame.available_annotation_types
                and AnnotationTypes.BoundingBoxes3D in self._annotation_types
                else None,
                "2": self._process_semantic_segmentation_2d(sensor_frame=camera_frame, fs_copy=True)
                if AnnotationTypes.SemanticSegmentation2D in camera_frame.available_annotation_types
                and AnnotationTypes.SemanticSegmentation2D in self._annotation_types
                else None,
                "4": self._process_instance_segmentation_2d(sensor_frame=camera_frame, fs_copy=True)
                if AnnotationTypes.InstanceSegmentation2D in camera_frame.available_annotation_types
                and AnnotationTypes.InstanceSegmentation2D in self._annotation_types
                else None,
                "6": self._process_depth(sensor_frame=camera_frame, fs_copy=True)
                if AnnotationTypes.Depth in camera_frame.available_annotation_types
                and AnnotationTypes.Depth in self._annotation_types
                else None,
                "8": self._process_motion_vectors_2d(sensor_frame=camera_frame, fs_copy=True)
                if AnnotationTypes.OpticalFlow in camera_frame.available_annotation_types
                and AnnotationTypes.OpticalFlow in self._annotation_types
                and not last_frame
                else None,
                "10": None,  # surface_normals_2d
            },
            sensor_data={
                "rgb": self._process_rgb(sensor_frame=camera_frame, fs_copy=True),
            },
        )


class DGPSummarizer(PartialsSummarizer):
    def summarize(self, partial_encodings: Dict[PartialEncoderName, Any]) -> Any:
        pass


class DGPDatasetEncoderV2(DatasetEncoderV2[SensorFrame[datetime]]):
    def __init__(
        self,
        dataset: Dataset,
        output_path: str,
        # summarizer: PartialsSummarizer = None,
        # encoders: List[PartialEncoder] = None,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
    ):
        # if encoders is None:
        #     encoders = [DGPCameraEncoder()]
        # if encoders is None:
        #     summarizer = DGPSummarizer()

        super().__init__(
            dataset=dataset,
            output_path=output_path,
            # encoders=encoders, summarizer=summarizer
        )

        if scene_names is not None:
            for sn in scene_names:
                if sn not in self._dataset.unordered_scene_names:
                    raise KeyError(f"{sn} could not be found in dataset {self._dataset.name}")
            self._scene_names = scene_names
        else:
            set_slice = slice(set_start, set_stop)
            self._scene_names = self._dataset.unordered_scene_names[set_slice]

    def sensor_frame_generator(self) -> Generator[SensorFrame[datetime], None, None]:
        for scene_name in self._scene_names:
            scene = self._dataset.get_scene(scene_name=scene_name)
            for sensor in scene.sensors:
                for frame_id in sensor.frame_ids:
                    sensor_frame = sensor.get_frame(frame_id=frame_id)
                    yield sensor_frame

    # def encode_dataset(self):
    #     source = self.sensor_frame_generator()
