from paralleldomain.decoding.decoder import Decoder
from paralleldomain.model.dataset import Dataset


def test_can_load_dataset_from_path(decoder: Decoder):
    dataset = decoder.get_dataset()

    assert len(dataset.scene_names) > 0


def test_can_load_scene(decoder: Decoder):
    dataset = decoder.get_dataset()

    scene = dataset.get_scene(scene_name=dataset.scene_names[0])
    assert scene is not None
    assert scene.name == dataset.scene_names[0]
