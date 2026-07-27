from __future__ import annotations

import json
import unittest

from my_agent.memory.evolver.maintenance.formal.tools import (
    formal_maintenance_tools,
    parse_maintenance_tool_call,
)
from my_agent.memory.experience.models import ExperienceTier
from my_agent.policy.identity import canonical_json_bytes
from my_agent.training.role_views import CanonicalToolCall


def _call(name: str, arguments: dict) -> CanonicalToolCall:
    return CanonicalToolCall("call-1", name, canonical_json_bytes(arguments).decode("utf-8"))


class FormalMaintenanceToolTests(unittest.TestCase):
    def test_tool_schema_is_fixed_and_has_no_promote(self) -> None:
        tools = formal_maintenance_tools()
        self.assertEqual(
            tuple(tool.name for tool in tools),
            ("lookup", "merge", "delete", "finish"),
        )
        lookup_schema = next(tool for tool in tools if tool.name == "lookup")
        parameters = json.loads(lookup_schema.parameters_json)
        self.assertEqual(
            tuple(parameters["properties"]["tiers"]["items"]["enum"]),
            tuple(tier.value for tier in ExperienceTier),
        )

    def test_parser_rejects_promote_and_extra_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_maintenance_tool_call((_call("promote", {"source_ids": ["tip-a"]}),))
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_maintenance_tool_call((_call("finish", {"summary": "done", "promote": True}),))

    def test_one_assistant_turn_must_issue_exactly_one_tool_call(self) -> None:
        finish = _call("finish", {"summary": "done"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_maintenance_tool_call(())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_maintenance_tool_call((finish, finish))

    def test_lookup_parser_rejects_invalid_tier_and_empty_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid experience tier"):
            parse_maintenance_tool_call((
                _call("lookup", {"query": "duplicates", "tiers": ["experience"]}),
            ))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            parse_maintenance_tool_call((_call("lookup", {"query": "", "tiers": []}),))


if __name__ == "__main__":
    unittest.main()
