"""_direction_records must keep every uploaded name that resolves to a hypothesis CURIE."""

from types import SimpleNamespace

from kestrel_backend.graph.nodes.synthesis import _direction_records
from kestrel_backend.graph.state import MemberWeight, ModuleSpine


def test_two_names_for_one_curie_contribute_all_signed_records():
    state = {
        "resolved_entities": [
            SimpleNamespace(curie="HGNC:11641", raw_name="TCF7L2", resolved_name="TCF7L2"),
            SimpleNamespace(curie="HGNC:11641", raw_name="transcription factor 7 like 2",
                            resolved_name="TCF7L2"),
        ],
        "module_spine": {
            "m": ModuleSpine(group="M", members={
                "TCF7L2": MemberWeight(name="TCF7L2", kme=0.8),
                "transcription factor 7 like 2": MemberWeight(
                    name="transcription factor 7 like 2", kme=-0.7),
            }),
        },
    }
    hyp = SimpleNamespace(supporting_entities=["HGNC:11641"])
    kmes = sorted(r[0] for r in _direction_records(hyp, state))
    assert kmes == [-0.7, 0.8]  # both signs present → direction can read as indeterminate
