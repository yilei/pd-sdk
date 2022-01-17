import argparse
import concurrent
import itertools
import logging
import os
import uuid
from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Generator, Generic, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from paralleldomain import Dataset
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.sensor import SensorFrame
from paralleldomain.model.type_aliases import SceneName, SensorName
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import relative_path
from paralleldomain.utilities.os import cpu_count

logger = logging.getLogger(__name__)

T = TypeVar("T")

_thread_pool_size = max(int(os.environ.get("ENCODER_THREAD_POOL_MAX_SIZE", cpu_count() * 4)), 4)


# class EncoderThreadPool(ThreadPoolExecutor):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.max_workers = self._max_workers
#
#     def map_async(self, fn: Callable[[Any], Any], iterable: Iterable[Any]) -> List[Future]:
#         return [self.submit(fn, i) for i in iterable]
#
#
# ENCODING_THREAD_POOL = EncoderThreadPool(_thread_pool_size)


class ObjectTransformer:
    @staticmethod
    def _filter_pre_transform(objects: Union[Generator, List]) -> Union[Generator, List]:
        return (o for o in objects)

    @staticmethod
    def _transform(objects: Union[Generator, List]) -> Union[Generator, List]:
        return (o for o in objects)

    @staticmethod
    def _filter_post_transform(objects: Union[Generator, List]) -> Union[Generator, List]:
        return (o for o in objects)

    @classmethod
    def transform(cls, objects: List) -> List:
        _pre_filtered = cls._filter_pre_transform(objects=objects)
        _transformed = cls._transform(objects=_pre_filtered)
        _post_filtered = cls._filter_post_transform(objects=_transformed)
        return list(_post_filtered)


class MaskTransformer:
    @staticmethod
    def _transform(mask: np.ndarray) -> np.ndarray:
        return mask

    @classmethod
    def transform(cls, mask: np.ndarray) -> np.ndarray:
        _transformed = cls._transform(mask)
        return _transformed


PartialEncoderName = str


class PartialEncoder(Generic[T]):
    @property
    @abstractmethod
    def name(self) -> PartialEncoderName:
        pass

    @abstractmethod
    def encode(self, item: T) -> Optional[Any]:
        pass


#
#
# class SceneEncoder(Generic[T]):
#     def __init__(
#         self,
#         dataset: Dataset,
#         scene_name: SceneName,
#         output_path: AnyPath,
#         encoders: List[PartialEncoder[T]]
#     ):
#         self._encoders = encoders
#         self._dataset: Dataset = dataset
#         self._scene_name: SceneName = scene_name
#         self._output_path: AnyPath = output_path
#         self._unordered_scene: UnorderedScene = dataset.get_unordered_scene(scene_name=scene_name)
#
#         self._camera_names: Optional[List[str]] = (
#             self._unordered_scene.camera_names if camera_names is None else camera_names
#         )
#         self._lidar_names: Optional[List[str]] = (
#             self._unordered_scene.lidar_names if lidar_names is None else lidar_names
#         )
#         self._frame_ids: Optional[List[str]] = self._unordered_scene.frame_ids if frame_ids is None else frame_ids
#         self._annotation_types: Optional[List[AnnotationType]] = (
#             self._unordered_scene.available_annotation_types if annotation_types is None else annotation_types
#         )
#
#         self._prepare_output_directories()
#
#     @abstractmethod
#     def source_generator(self) -> Union[Generator[T, None, None], List[T]]:
#         pass
#
#
#
#     @property
#     def _sensor_names(self) -> List[str]:
#         return self._camera_names + self._lidar_names
#
#     def _relative_path(self, path: AnyPath) -> AnyPath:
#         return relative_path(path, self._output_path)
#
#     def _run_async(self, func: Callable[[Any], Any], *args, **kwargs) -> Future:
#         return ENCODING_THREAD_POOL.submit(func, *args, **kwargs)
#
#     def _prepare_output_directories(self) -> None:
#         if not self._output_path.is_cloud_path:
#             self._output_path.mkdir(exist_ok=True, parents=True)
#
#     @abstractmethod
#     def _encode_camera_frame(self, camera_name: str, camera_frame: SensorFrame):
#         ...
#
#     @abstractmethod
#     def _encode_lidar_frame(self, frame_id: str, lidar_frame: SensorFrame):
#         ...
#
#     def _encode_cameras(self) -> Any:
#         return [self._encode_camera(camera_name=c).result() for c in self._camera_names]
#
#     def _encode_camera(self, camera_name: SensorName) -> Future:
#         frame_ids = self._frame_ids
#         camera_encoding_futures = {
#             ENCODING_THREAD_POOL.submit(
#                 lambda fid: self._encode_camera_frame(
#                     camera_name=camera_name,
#                     camera_frame=self._unordered_scene.get_frame(frame_id=fid).get_sensor(sensor_name=camera_name),
#                 ),
#                 frame_id,
#             )
#             for frame_id in frame_ids
#         }
#         return ENCODING_THREAD_POOL.submit(lambda: concurrent.futures.wait(camera_encoding_futures))
#
#     def _encode_lidar(self, lidar_name: SensorName) -> Future:
#         frame_ids = self._frame_ids
#         lidar_encoding_futures = {
#             ENCODING_THREAD_POOL.submit(
#                 lambda fid: self._encode_lidar_frame(
#                     frame_id=frame_id,
#                     lidar_frame=self._unordered_scene.get_frame(frame_id=fid).get_sensor(sensor_name=lidar_name),
#                 ),
#                 frame_id,
#             )
#             for frame_id in frame_ids
#         }
#         return ENCODING_THREAD_POOL.submit(lambda: concurrent.futures.wait(lidar_encoding_futures))
#
#     def _encode_lidars(self) -> Any:
#         return [self._encode_lidar(lidar_name=ln).result() for ln in self._lidar_names]
#
#     def _encode_sensors(self):
#         self._encode_cameras()
#         self._encode_lidars()
#
#     def _run_encoding(self) -> Any:
#         self._encode_sensors()
#         logger.info(f"Successfully encoded {self._scene_name}")
#         return str(uuid.uuid4())
#
#     def encode_scene(self) -> Any:
#         for source_item in self.source_generator():
#             for encoder in self._encoders:
#                 encoder.encode(objects=source_item)
#         encoding_result = self._run_encoding()
#         return encoding_result


class DatasetEncoderV2(Generic[T]):
    def __init__(
        self,
        dataset: Dataset,
        output_path: str,
        # encoders: List[PartialEncoder[T]],
        # summarizer: PartialsSummarizer,
    ):
        # self.summarizer = summarizer
        # self._encoders = encoders
        self._dataset = dataset
        self._output_path = AnyPath(output_path)

    # @abstractmethod
    # def encoding_item_generator(self) -> Generator[T, None, None]:
    #     pass

    @abstractmethod
    def encode_dataset(self):
        pass
        # partial_encodings = {e.name: list() for e in self._encoders}
        # for source_item in self.encoding_item_generator():
        #     for encoder in self._encoders:
        #         partial_encoded = encoder.encode(item=source_item)
        #         if partial_encoded is not None:
        #             partial_encodings[encoder.name].append(partial_encoded)
        # self.summarizer.summarize(partial_encodings=partial_encodings)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        output_path: str,
        # encoders: List[PartialEncoder[T]],
        # summarizer: PartialsSummarizer,
    ) -> "DatasetEncoderV2[T]":
        return cls(
            dataset=dataset,
            output_path=output_path,
            # encoders=encoders, summarizer=summarizer
        )
