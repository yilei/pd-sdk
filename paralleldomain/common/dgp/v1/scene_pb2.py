# flake8: noqa
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: scene.proto

import sys

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode("latin1"))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2

import paralleldomain.common.dgp.v1.sample_pb2 as sample__pb2
import paralleldomain.common.dgp.v1.statistics_pb2 as statistics__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name="scene.proto",
    package="dgp.proto",
    syntax="proto3",
    serialized_options=None,
    serialized_pb=_b(
        '\n\x0bscene.proto\x12\tdgp.proto\x1a\x0csample.proto\x1a\x10statistics.proto\x1a\x19google/protobuf/any.proto\x1a\x1fgoogle/protobuf/timestamp.proto"\xc2\x03\n\x05Scene\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12\x0b\n\x03log\x18\x03 \x01(\t\x12"\n\x07samples\x18\x04 \x03(\x0b\x32\x11.dgp.proto.Sample\x12\x30\n\x08metadata\x18\x05 \x03(\x0b\x32\x1e.dgp.proto.Scene.MetadataEntry\x12\x1e\n\x04\x64\x61ta\x18\x06 \x03(\x0b\x32\x10.dgp.proto.Datum\x12\x31\n\rcreation_date\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x34\n\nontologies\x18\x08 \x03(\x0b\x32 .dgp.proto.Scene.OntologiesEntry\x12\x30\n\nstatistics\x18\t \x01(\x0b\x32\x1c.dgp.proto.DatasetStatistics\x1a\x45\n\rMetadataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Any:\x02\x38\x01\x1a\x31\n\x0fOntologiesEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01"*\n\x06Scenes\x12 \n\x06scenes\x18\x01 \x03(\x0b\x32\x10.dgp.proto.Scene"\x1f\n\nSceneFiles\x12\x11\n\tfilenames\x18\x01 \x03(\tb\x06proto3'
    ),
    dependencies=[
        sample__pb2.DESCRIPTOR,
        statistics__pb2.DESCRIPTOR,
        google_dot_protobuf_dot_any__pb2.DESCRIPTOR,
        google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,
    ],
)


_SCENE_METADATAENTRY = _descriptor.Descriptor(
    name="MetadataEntry",
    full_name="dgp.proto.Scene.MetadataEntry",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="key",
            full_name="dgp.proto.Scene.MetadataEntry.key",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="value",
            full_name="dgp.proto.Scene.MetadataEntry.value",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=_b("8\001"),
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=449,
    serialized_end=518,
)

_SCENE_ONTOLOGIESENTRY = _descriptor.Descriptor(
    name="OntologiesEntry",
    full_name="dgp.proto.Scene.OntologiesEntry",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="key",
            full_name="dgp.proto.Scene.OntologiesEntry.key",
            index=0,
            number=1,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="value",
            full_name="dgp.proto.Scene.OntologiesEntry.value",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=_b("8\001"),
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=520,
    serialized_end=569,
)

_SCENE = _descriptor.Descriptor(
    name="Scene",
    full_name="dgp.proto.Scene",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="name",
            full_name="dgp.proto.Scene.name",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="description",
            full_name="dgp.proto.Scene.description",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="log",
            full_name="dgp.proto.Scene.log",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="samples",
            full_name="dgp.proto.Scene.samples",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="metadata",
            full_name="dgp.proto.Scene.metadata",
            index=4,
            number=5,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="data",
            full_name="dgp.proto.Scene.data",
            index=5,
            number=6,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="creation_date",
            full_name="dgp.proto.Scene.creation_date",
            index=6,
            number=7,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="ontologies",
            full_name="dgp.proto.Scene.ontologies",
            index=7,
            number=8,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="statistics",
            full_name="dgp.proto.Scene.statistics",
            index=8,
            number=9,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[
        _SCENE_METADATAENTRY,
        _SCENE_ONTOLOGIESENTRY,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=119,
    serialized_end=569,
)


_SCENES = _descriptor.Descriptor(
    name="Scenes",
    full_name="dgp.proto.Scenes",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="scenes",
            full_name="dgp.proto.Scenes.scenes",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=571,
    serialized_end=613,
)


_SCENEFILES = _descriptor.Descriptor(
    name="SceneFiles",
    full_name="dgp.proto.SceneFiles",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="filenames",
            full_name="dgp.proto.SceneFiles.filenames",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=615,
    serialized_end=646,
)

_SCENE_METADATAENTRY.fields_by_name["value"].message_type = google_dot_protobuf_dot_any__pb2._ANY
_SCENE_METADATAENTRY.containing_type = _SCENE
_SCENE_ONTOLOGIESENTRY.containing_type = _SCENE
_SCENE.fields_by_name["samples"].message_type = sample__pb2._SAMPLE
_SCENE.fields_by_name["metadata"].message_type = _SCENE_METADATAENTRY
_SCENE.fields_by_name["data"].message_type = sample__pb2._DATUM
_SCENE.fields_by_name["creation_date"].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_SCENE.fields_by_name["ontologies"].message_type = _SCENE_ONTOLOGIESENTRY
_SCENE.fields_by_name["statistics"].message_type = statistics__pb2._DATASETSTATISTICS
_SCENES.fields_by_name["scenes"].message_type = _SCENE
DESCRIPTOR.message_types_by_name["Scene"] = _SCENE
DESCRIPTOR.message_types_by_name["Scenes"] = _SCENES
DESCRIPTOR.message_types_by_name["SceneFiles"] = _SCENEFILES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Scene = _reflection.GeneratedProtocolMessageType(
    "Scene",
    (_message.Message,),
    dict(
        MetadataEntry=_reflection.GeneratedProtocolMessageType(
            "MetadataEntry",
            (_message.Message,),
            dict(
                DESCRIPTOR=_SCENE_METADATAENTRY,
                __module__="scene_pb2"
                # @@protoc_insertion_point(class_scope:dgp.proto.Scene.MetadataEntry)
            ),
        ),
        OntologiesEntry=_reflection.GeneratedProtocolMessageType(
            "OntologiesEntry",
            (_message.Message,),
            dict(
                DESCRIPTOR=_SCENE_ONTOLOGIESENTRY,
                __module__="scene_pb2"
                # @@protoc_insertion_point(class_scope:dgp.proto.Scene.OntologiesEntry)
            ),
        ),
        DESCRIPTOR=_SCENE,
        __module__="scene_pb2"
        # @@protoc_insertion_point(class_scope:dgp.proto.Scene)
    ),
)
_sym_db.RegisterMessage(Scene)
_sym_db.RegisterMessage(Scene.MetadataEntry)
_sym_db.RegisterMessage(Scene.OntologiesEntry)

Scenes = _reflection.GeneratedProtocolMessageType(
    "Scenes",
    (_message.Message,),
    dict(
        DESCRIPTOR=_SCENES,
        __module__="scene_pb2"
        # @@protoc_insertion_point(class_scope:dgp.proto.Scenes)
    ),
)
_sym_db.RegisterMessage(Scenes)

SceneFiles = _reflection.GeneratedProtocolMessageType(
    "SceneFiles",
    (_message.Message,),
    dict(
        DESCRIPTOR=_SCENEFILES,
        __module__="scene_pb2"
        # @@protoc_insertion_point(class_scope:dgp.proto.SceneFiles)
    ),
)
_sym_db.RegisterMessage(SceneFiles)


_SCENE_METADATAENTRY._options = None
_SCENE_ONTOLOGIESENTRY._options = None
# @@protoc_insertion_point(module_scope)
