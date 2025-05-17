import os
import shutil
import json
import pytest
from claude_testing_v1 import run_claude_json


CLAUDE_AVAILABLE = shutil.which("claude") is not None

skip_reason = "claude CLI not available or RUN_CLAUDE_TESTS not set"

@pytest.mark.skipif(not CLAUDE_AVAILABLE or os.environ.get("RUN_CLAUDE_TESTS") != "1", reason=skip_reason)
def test_run_claude_json():
    """Basic validation that Claude returns valid JSON using --output-format."""
    output = run_claude_json("hello", allowed_tools=["Bash"])
    assert isinstance(output, dict)
    assert output
