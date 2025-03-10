from typing import Dict

import numpy as np

from paralleldomain import Dataset, Scene
from paralleldomain.model.annotation import (
    AnnotationTypes,
    BoundingBox2D,
    BoundingBox3D,
    PointCache,
    PointCacheComponent,
)


class TestSensorFrame:
    def test_box_3d_loading(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        lidar_sensor = next(iter(frame.lidar_frames))
        boxes = lidar_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)

        assert isinstance(boxes.boxes, list)
        assert len(boxes.boxes) > 0

        for box in boxes.boxes:
            assert isinstance(box, BoundingBox3D)
            assert isinstance(box.pose.translation, np.ndarray)
            assert isinstance(box.pose.transformation_matrix, np.ndarray)
            assert isinstance(box.class_id, int)

    def test_point_cache_loading(self, scene: Scene):
        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])
        lidar_sensor = next(iter(frame.lidar_frames))
        assert AnnotationTypes.PointCaches in lidar_sensor.available_annotation_types
        point_caches = lidar_sensor.get_annotations(annotation_type=AnnotationTypes.PointCaches)

        assert isinstance(point_caches.caches, list)
        assert len(point_caches.caches) > 0

        for cache in point_caches.caches:
            assert isinstance(cache, PointCache)
            assert len(cache.components) > 0
            assert isinstance(cache.instance_id, int)

            for component in cache.components:
                assert isinstance(component, PointCacheComponent)
                assert isinstance(component.points, np.ndarray)
                assert isinstance(component.normals, np.ndarray)

    def test_box_2d_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.BoundingBoxes2D in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter([f for f in frame.camera_frames if "virtual" not in f.sensor_name]))
        boxes = camera_sensor.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)

        assert isinstance(boxes.boxes, list)
        assert len(boxes.boxes) > 0

        for box in boxes.boxes:
            assert isinstance(box, BoundingBox2D)
            assert isinstance(box.x, int)
            assert isinstance(box.y, int)
            assert isinstance(box.width, int)
            assert isinstance(box.height, int)
            assert isinstance(box.attributes, Dict)
            assert isinstance(box.class_id, int)

    def test_instance_seg_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.InstanceSegmentation2D in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))
        id_mask = camera_sensor.get_annotations(annotation_type=AnnotationTypes.InstanceSegmentation2D)

        assert id_mask is not None
        instance_ids = id_mask.instance_ids
        assert instance_ids.shape[2] == 1
        assert len(instance_ids.shape) == 3

        image = camera_sensor.image.rgb
        assert image.shape[:2] == instance_ids.shape[:2]

    def test_sem_seg_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.SemanticSegmentation2D in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))
        semseg = camera_sensor.get_annotations(annotation_type=AnnotationTypes.SemanticSegmentation2D)

        assert semseg is not None
        class_ids = semseg.class_ids
        assert class_ids.shape[2] == 1
        assert len(class_ids.shape) == 3

        image = camera_sensor.image.rgb
        assert image.shape[:2] == class_ids.shape[:2]

    def test_optical_flow_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.OpticalFlow in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])

        camera_sensor = next(
            (x for x in frame.camera_frames if AnnotationTypes.OpticalFlow in x.available_annotation_types), None
        )
        assert camera_sensor is not None

        flow = camera_sensor.get_annotations(annotation_type=AnnotationTypes.OpticalFlow)
        assert flow is not None
        image = camera_sensor.image.rgb
        assert flow.vectors.shape[2] == 2
        assert image.shape[:2] == flow.vectors.shape[:2]

    def test_scene_flow_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.SceneFlow in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])

        lidar_sensor = next(
            (x for x in frame.lidar_frames if AnnotationTypes.SceneFlow in x.available_annotation_types), None
        )
        assert lidar_sensor is not None

        flow = lidar_sensor.get_annotations(annotation_type=AnnotationTypes.SceneFlow)
        assert flow is not None
        cloud = lidar_sensor.point_cloud.xyz
        assert flow.vectors.shape[0] == cloud.shape[0]
        assert flow.vectors.shape[1] == 3

    def test_surface_normals_3d_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.SurfaceNormals3D in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])

        lidar_sensor = next(
            (x for x in frame.lidar_frames if AnnotationTypes.SurfaceNormals3D in x.available_annotation_types), None
        )
        assert lidar_sensor is not None
        normals = lidar_sensor.get_annotations(annotation_type=AnnotationTypes.SurfaceNormals3D)
        assert normals is not None
        cloud = lidar_sensor.point_cloud.xyz
        assert normals.normals.shape[0] == cloud.shape[0]
        assert normals.normals.shape[1] == 3

    def test_surface_normals_2d_loading(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.SurfaceNormals2D in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[0])

        camera_sensor = next(
            (x for x in frame.camera_frames if AnnotationTypes.OpticalFlow in x.available_annotation_types), None
        )
        assert camera_sensor is not None

        normals = camera_sensor.get_annotations(annotation_type=AnnotationTypes.SurfaceNormals2D)
        assert normals is not None
        image = camera_sensor.image.rgb
        assert normals.normals.shape[:2] == image.shape[:2]
        assert normals.normals.shape[2] == 3

        assert np.allclose(np.linalg.norm(normals.normals, axis=-1), 1.0)

    def test_image_coordinates(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.OpticalFlow in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[-1])
        camera_sensor = next(iter(frame.camera_frames))

        rgb = camera_sensor.image.rgb
        coordinates = camera_sensor.image.coordinates
        for y in range(rgb.shape[0]):
            for x in range(rgb.shape[1]):
                assert np.all(coordinates[y, x] == np.array([y, x]))

    """
    @pytest.skip
    def test_image_warp(self, scene: Scene, dataset: Dataset):
        assert AnnotationTypes.OpticalFlow in dataset.available_annotation_types

        frame_ids = scene.frame_ids
        frame = scene.get_frame(frame_id=frame_ids[5])
        camera_sensor = next(iter(frame.camera_frames))
        flow = camera_sensor.get_annotations(
            annotation_type=AnnotationTypes.OpticalFlow
        )


        rgb = camera_sensor.image.rgb
        next_image = np.zeros_like(rgb)
        coordinates = camera_sensor.image.coordinates
        next_frame_coords = coordinates + flow.vectors

        for y in range(rgb.shape[0]):
            for x in range(rgb.shape[1]):
                next_coord = next_frame_coords[y, x]
                if 0 <= next_coord[0] < rgb.shape[0] and 0 <= next_coord[1] < rgb.shape[1]:
                    next_image[next_coord[0], next_coord[1], :] = rgb[y, x, :]

        import cv2
        cv2.imshow("window_name", next_image[..., [2,1,0]])
        cv2.waitKey()
    """
