import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import (
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
)
from agent import FYPResearchAgent
from llm_client import LLMClient
from model_contracts import classify_model_error


def fake_client(*, response=None, error=None):
    create = Mock()
    if error is not None:
        create.side_effect = error
    else:
        create.return_value = response
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def successful_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Structured model answer."),
        )],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
        ),
    )


class ModelContractTests(unittest.TestCase):
    def test_success_has_stable_contract_and_compatibility_latency(self):
        provider = fake_client(response=successful_response())
        client = LLMClient(client=provider, model="test-model")

        result = client.generate("system", "user")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], "Structured model answer.")
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["usage"]["total_tokens"], 20)
        self.assertIsNone(result["error"])
        self.assertEqual(result["latency"], result["latency_seconds"])
        provider.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            temperature=0.3,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
        )

    def test_timeout_returns_error_result_instead_of_raising(self):
        client = LLMClient(
            client=fake_client(error=TimeoutError("provider secret detail")),
            model="test-model",
        )

        result = client.generate("system", "user")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "model_timeout")
        self.assertTrue(result["error"]["retryable"])
        self.assertNotIn("provider secret detail", str(result))
        self.assertIsNone(result["usage"]["total_tokens"])

    def test_error_classifier_distinguishes_failure_categories(self):
        class ProviderStatusError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        cases = [
            (ConnectionError("offline"), "model_connection_error", True),
            (ProviderStatusError(429), "model_rate_limited", True),
            (ProviderStatusError(401), "model_authentication_error", False),
            (ProviderStatusError(503), "model_provider_error", True),
            (ValueError("bad response"), "model_call_error", False),
        ]

        for error, expected_code, retryable in cases:
            with self.subTest(expected_code=expected_code):
                result = classify_model_error(error)
                self.assertEqual(result.code, expected_code)
                self.assertEqual(result.retryable, retryable)

    def test_client_configures_one_finite_sdk_retry_policy(self):
        with (
            patch("llm_client.validate_config") as validate,
            patch("llm_client.OpenAI") as openai_client,
        ):
            client = LLMClient(model="test-model")

        validate.assert_called_once_with()
        openai_client.assert_called_once()
        kwargs = openai_client.call_args.kwargs
        self.assertEqual(kwargs["timeout"], LLM_TIMEOUT_SECONDS)
        self.assertEqual(kwargs["max_retries"], LLM_MAX_RETRIES)
        self.assertEqual(client.max_retries, LLM_MAX_RETRIES)
        self.assertEqual(client.max_output_tokens, LLM_MAX_OUTPUT_TOKENS)

    def test_unexpected_response_shape_is_also_contained(self):
        client = LLMClient(
            client=fake_client(response=SimpleNamespace(
                choices=[],
                usage=None,
            )),
            model="test-model",
        )

        result = client.generate("system", "user")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "model_call_error")
        self.assertIn("模型服务当前不可用", result["content"])

    def test_full_agent_returns_model_failure_without_recording_fake_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = FYPResearchAgent(
                memory_store_path=str(Path(directory) / "memories.json")
            )
            agent.llm = LLMClient(
                client=fake_client(error=ConnectionError("offline")),
                model="test-model",
            )

            result = agent.run("解释 Transformer")

        self.assertEqual(result["model_call"]["status"], "error")
        self.assertEqual(
            result["model_call"]["error"]["code"],
            "model_connection_error",
        )
        self.assertIn("模型服务当前不可用", result["response"])
        self.assertIsNone(result["usage"]["total_tokens"])
        self.assertEqual(agent.conversation_memory.get_recent_history(), [])


if __name__ == "__main__":
    unittest.main()
