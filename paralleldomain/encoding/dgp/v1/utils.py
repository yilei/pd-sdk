from typing import Dict, List

import ujson

from paralleldomain.common.dgp.v1 import ontology_pb2
from paralleldomain.model.class_mapping import ClassMap


def class_map_to_ontology_proto(class_map: ClassMap):
    return ontology_pb2.Ontology(
        items=[
            ontology_pb2.OntologyItem(
                id=cid,
                name=cval.name,
                color=ontology_pb2.OntologyItem.Color(
                    r=cval.meta["color"]["r"],
                    g=cval.meta["color"]["g"],
                    b=cval.meta["color"]["b"],
                ),
                isthing=cval.instanced,
                supercategory="",
            )
            for cid, cval in class_map.items()
        ]
    )


def _attribute_key_dump(obj: object) -> str:
    return str(obj)


def _attribute_value_dump(obj: object) -> str:
    if isinstance(obj, Dict) or isinstance(obj, List):
        return ujson.dumps(obj, indent=2, escape_forward_slashes=False)
    else:
        return str(obj)
