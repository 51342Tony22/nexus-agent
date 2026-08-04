# llm_client.py
#调用api


from time import perf_counter

from openai import OpenAI
from config import (
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL,
    validate_config,
)
from model_contracts import (
    ModelCallResult,
    TokenUsage,
    classify_model_error,
)

class LLMClient:
    def __init__(self, client=None, model: str | None = None):
        self.model = model or SILICONFLOW_MODEL
        self.timeout_seconds = LLM_TIMEOUT_SECONDS
        self.max_retries = LLM_MAX_RETRIES
        self.max_output_tokens = LLM_MAX_OUTPUT_TOKENS

        if client is not None:
            # Dependency injection lets the contract be tested without a
            # network request or real provider credentials.
            self.client = client
            return

        validate_config()
        self.client = OpenAI(
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        started_at = perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.3,
                max_tokens=self.max_output_tokens,
            )
            latency = perf_counter() - started_at
            usage = response.usage
            return ModelCallResult(
                status="success",
                content=response.choices[0].message.content or "",
                model=self.model,
                latency_seconds=latency,
                usage=TokenUsage(
                    prompt_tokens=(
                        usage.prompt_tokens if usage else None
                    ),
                    completion_tokens=(
                        usage.completion_tokens if usage else None
                    ),
                    total_tokens=(usage.total_tokens if usage else None),
                ),
                error=None,
                timeout_seconds=self.timeout_seconds,
                max_retries_configured=self.max_retries,
                max_output_tokens_configured=self.max_output_tokens,
            ).to_runtime_dict()
        except Exception as error:
            latency = perf_counter() - started_at
            model_error = classify_model_error(error)
            return ModelCallResult(
                status="error",
                content=(
                    "模型服务当前不可用，Agent 已安全停止本轮生成。"
                    f"错误代码：{model_error.code}。"
                ),
                model=self.model,
                latency_seconds=latency,
                usage=TokenUsage(),
                error=model_error,
                timeout_seconds=self.timeout_seconds,
                max_retries_configured=self.max_retries,
                max_output_tokens_configured=self.max_output_tokens,
            ).to_runtime_dict()
