from __future__ import annotations

from finpilot.agent.prompts import build_agent_decision_prompt


def test_agent_decision_prompt_requires_tool_args_to_match_selected_tool_arguments():
    prompt = build_agent_decision_prompt(
        {
            "sub_agent": {
                "allowed_tools": [
                    {
                        "name": "transfer_mock_funds",
                        "arguments": {"account_no": "string", "amount": "number", "currency": "string, optional"},
                    }
                ]
            }
        }
    )

    assert "tool_args must match the selected tool's arguments" in prompt
