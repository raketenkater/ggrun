"""Guards for the trusted repair oracle and the real tool loop."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('agent_workload', Path(__file__).resolve().parents[1] / 'scripts/verify-agent-workload.py')
work = importlib.util.module_from_spec(spec)
spec.loader.exec_module(work)

class AgentWorkloadTests(unittest.TestCase):
    def test_correct_repairs_and_broken_inputs(self):
        repairs = ['def solve(n, d):\n return (n + d - 1) // d',
                   'def solve(x, low, high):\n return max(low, min(x, high))',
                   'def solve(a, b, c, d):\n return max(0, min(b, d) - max(a, c))']
        for task, repair in zip(work.TASKS, repairs):
            self.assertFalse(work.oracle(task['source'], task)['passed'])
            self.assertTrue(work.oracle(repair, task)['passed'])

    def test_model_code_cannot_execute_or_change_oracle(self):
        for source in ["import os\ndef solve(n, d): return 0",
                       "def solve(n, d): return __import__('os').system('id')",
                       "def solve(n, d): return n ** 99999999",
                       "def solve(n, d): return (1).__class__",
                       "def solve(n, d): return solve(n, d)"]:
            self.assertFalse(work.oracle(source, work.TASKS[0])['passed'])

    def test_claimed_completion_without_tools_is_not_success(self):
        args = SimpleNamespace(max_turns=2, url='http://unused', model='local', seed=1, max_tokens=100, timeout=1)
        with tempfile.TemporaryDirectory() as temp, patch.object(work, 'http_json', return_value={
                'choices': [{'message': {'role': 'assistant', 'content': 'All tests pass.'}}]}):
            result = work.run_task(args, work.TASKS[0], 0, Path(temp))
        self.assertFalse(result['passed'])

    def test_repair_requires_test_after_last_write(self):
        import json
        def response(calls):
            return {'choices': [{'message': {'role': 'assistant', 'content': None,
                    'tool_calls': [{'id': str(i), 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(data)}}
                                   for i, (name, data) in enumerate(calls)]}}]}
        calls = [('read_file', {'path': 'solution.py'}), ('run_tests', {}),
                 ('write_file', {'path': 'solution.py', 'content': 'def solve(n, d):\n return (n+d-1)//d'})]
        args = SimpleNamespace(max_turns=1, url='http://unused', model='local', seed=1, max_tokens=100, timeout=1)
        for rerun in (False, True):
            with tempfile.TemporaryDirectory() as temp, patch.object(work, 'http_json', return_value=response(calls + ([('run_tests', {})] if rerun else []))):
                result = work.run_task(args, work.TASKS[0], 0, Path(temp))
            self.assertEqual(result['passed'], rerun)

if __name__ == '__main__': unittest.main()
