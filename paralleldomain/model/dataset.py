import contextlib
from dataclasses import dataclass, field
from typing import Any, ContextManager, Dict, Generic, List, Set, TypeVar

from paralleldomain.model.sensor_frame_set import SensorFrameSet, SensorFrameSetDecoderProtocol

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

import logging

from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.scene import Scene, SceneDecoderProtocol
from paralleldomain.model.type_aliases import SceneName, SensorFrameSetName

logger = logging.getLogger(__name__)


@dataclass
class DatasetMeta:
    """*Dataclass*

    Stores name, annotation types and any custom meta attributes for a dataset"""

    name: str
    """Dataset name."""
    available_annotation_types: List[AnnotationType]
    """List of all available annotation types in this dataset."""
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    """Dictionary of custom attributes for the dataset."""


class DatasetDecoderProtocol(SensorFrameSetDecoderProtocol, Protocol):
    """Interface Definition for decoder implementations.

    Not to be instantiated directly!
    """

    def get_sensor_frame_set_names(self) -> Set[SensorFrameSetName]:
        pass

    def get_sensor_frame_set(self, set_name: SensorFrameSetName) -> SensorFrameSet:
        pass

    def get_dataset_meta_data(self) -> DatasetMeta:
        pass


class SceneDatasetDecoderProtocol(DatasetDecoderProtocol, SceneDecoderProtocol, Protocol):
    """Interface Definition for decoder implementations.

    Not to be instantiated directly!
    """

    def get_scene_names(self) -> Set[SceneName]:
        pass

    def get_scene(self, scene_name: SceneName) -> Scene:
        pass

    def get_dataset_meta_data(self) -> DatasetMeta:
        pass


class Dataset:
    """The :obj:`Dataset` object is the entry point for loading any data.

    A dataset manages all attached scenes and its sensor data. It takes care of calling the decoder when specific
    data is required and stores it in the PD SDK model classes and attributes.
    """

    def __init__(self, decoder: DatasetDecoderProtocol):
        self._decoder = decoder

    @property
    def sensor_frame_set_names(self) -> Set[SensorFrameSetName]:
        """Returns a list of sensor frame set names within the dataset."""
        return self._decoder.get_sensor_frame_set_names()

    @property
    def meta_data(self) -> DatasetMeta:
        """Returns a list of scene names within the dataset."""
        return self._decoder.get_dataset_meta_data()

    @property
    def sensor_frame_sets(self) -> Dict[SensorFrameSetName, SensorFrameSet]:
        """Returns a dictionary of :obj:`SensorFrameSet` instances with the scene name as key."""
        return {name: self._decoder.get_sensor_frame_set(set_name=name) for name in self.sensor_frame_set_names}

    @property
    def available_annotation_types(self) -> List[AnnotationType]:
        """Returns a list of available annotation types for the dataset."""
        return self.meta_data.available_annotation_types

    @property
    def name(self) -> str:
        """Returns the name of the dataset."""
        return self.meta_data.name

    def get_sensor_frame_set(self, set_name: SensorFrameSetName) -> SensorFrameSet:
        """Allows access to a sensor frame set by using its name.

        Args:
            set_name (str): Name of sensor frame set to be returned

        Returns:
            Returns the `SensorFrameSet` object for a sensor frame set name.
        """
        return self._decoder.get_sensor_frame_set(set_name=set_name)


class SceneDataset(Dataset):
    """The :obj:`Dataset` object is the entry point for loading any data.

    A dataset manages all attached scenes and its sensor data. It takes care of calling the decoder when specific
    data is required and stores it in the PD SDK model classes and attributes.
    """

    def __init__(self, decoder: SceneDatasetDecoderProtocol):
        super().__init__(decoder=decoder)
        self._decoder = decoder

    @property
    def scene_names(self) -> Set[SceneName]:
        """Returns a list of scene names within the dataset."""
        return self._decoder.get_scene_names()

    @property
    def scenes(self) -> Dict[SceneName, Scene]:
        """Returns a dictionary of :obj:`Scene` instances with the scene name as key."""
        return {name: self._decoder.get_scene(scene_name=name) for name in self.scene_names}

    def get_scene(self, scene_name: SceneName) -> Scene:
        """Allows access to a scene by using its name.

        Args:
            scene_name (str): Name of scene to be returned

        Returns:
            Returns the `Scene` object for a scene name.
        """
        return self._decoder.get_scene(scene_name=scene_name)
