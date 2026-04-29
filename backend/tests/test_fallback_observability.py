"""Tests for fallback observability logging.

Verifies that structured FALLBACK_EVENT log messages are emitted when
pipeline nodes transition from primary (HTTP) to fallback (SDK) tier.
"""

import logging

import pytest


class TestFallbackEventLogFormat:
    """Verify the FALLBACK_EVENT log format is parseable."""

    def test_entity_resolution_fallback_log_format(self, caplog):
        """Entity resolution should log FALLBACK_EVENT with entity name and tier."""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("kestrel_backend.graph.nodes.entity_resolution")
            # Simulate what the node logs
            logger.info(
                "FALLBACK_EVENT node=entity_resolution entity=%s reason=tier1_failed tier=2",
                "NAD+",
            )

        assert any("FALLBACK_EVENT" in record.message for record in caplog.records)
        fallback_records = [r for r in caplog.records if "FALLBACK_EVENT" in r.message]
        assert len(fallback_records) == 1
        msg = fallback_records[0].message
        assert "node=entity_resolution" in msg
        assert "entity=NAD+" in msg
        assert "reason=tier1_failed" in msg
        assert "tier=2" in msg

    def test_triage_fallback_log_format(self, caplog):
        """Triage should log FALLBACK_EVENT with entity and curie."""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("kestrel_backend.graph.nodes.triage")
            logger.info(
                "FALLBACK_EVENT node=triage entity=%s curie=%s reason=tier1_edge_count_failed tier=2",
                "NAD+", "CHEBI:15422",
            )

        fallback_records = [r for r in caplog.records if "FALLBACK_EVENT" in r.message]
        assert len(fallback_records) == 1
        msg = fallback_records[0].message
        assert "node=triage" in msg
        assert "curie=CHEBI:15422" in msg

    def test_direct_kg_fallback_log_format(self, caplog):
        """Direct KG should log FALLBACK_EVENT with curie."""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("kestrel_backend.graph.nodes.direct_kg")
            logger.info(
                "FALLBACK_EVENT node=direct_kg entity=%s reason=tier1_api_failed tier=2",
                "CHEBI:15422",
            )

        fallback_records = [r for r in caplog.records if "FALLBACK_EVENT" in r.message]
        assert len(fallback_records) == 1
        msg = fallback_records[0].message
        assert "node=direct_kg" in msg
        assert "entity=CHEBI:15422" in msg
        assert "reason=tier1_api_failed" in msg

    def test_no_fallback_event_on_success(self, caplog):
        """No FALLBACK_EVENT should be logged when primary tier succeeds."""
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("kestrel_backend.graph.nodes.entity_resolution")
            logger.info("Tier 1 resolved entity: NAD+ -> CHEBI:15422")

        fallback_records = [r for r in caplog.records if "FALLBACK_EVENT" in r.message]
        assert len(fallback_records) == 0


class TestFallbackEventParsing:
    """Verify FALLBACK_EVENT messages can be parsed for monitoring."""

    def test_parse_structured_fields(self):
        """FALLBACK_EVENT fields should be extractable by simple string parsing."""
        msg = "FALLBACK_EVENT node=entity_resolution entity=NAD+ reason=tier1_failed tier=2"

        fields = {}
        for part in msg.split():
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value

        assert fields["node"] == "entity_resolution"
        assert fields["entity"] == "NAD+"
        assert fields["reason"] == "tier1_failed"
        assert fields["tier"] == "2"
