import logging
import time
from typing import List, Dict, Optional

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from configs.settings import settings

# Configure a dedicated logger for this module for better observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Global LLM Client Variable ---
llm_client: Optional['EnterpriseMistralClient'] = None


class EnterpriseMistralClient:
    """
    An enterprise-grade client for a more stable interaction with Mistral's API.
    """
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("Mistral API key cannot be None or empty.")
        logger.info(f"Initializing EnterpriseMistralClient for model: {model_name}")
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name
        logger.info("Mistral client initialized successfully.")

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def invoke_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False
    ) -> Optional[str]:
        """
        Invokes the chat completion endpoint with retry logic and explicit message typing.
        """
        logger.debug(f"Invoking chat completion for model {self.model_name} with {len(messages)} messages.")
        start_time = time.time()

        try:
            # Validate and prepare messages as required by the latest SDK (list of dicts)
            chat_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages if "role" in msg and "content" in msg
            ]

            if not chat_messages:
                logger.warning("No valid messages to send.")
                return None

            response_format = {"type": "json_object"} if json_mode else None

            # Correct async API call for mistralai >= 1.8.x
            response = await self.client.chat.complete_async(
                model=self.model_name,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )

            duration = time.time() - start_time
            usage_info = f"Usage: {getattr(response, 'usage', 'N/A')}"
            logger.info(f"Chat completion successful. Duration: {duration:.2f}s. {usage_info}")

            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                logger.warning("Received a response with no choices.")
                return None

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Chat completion failed after {duration:.2f}s. Error: {e}")
            raise

def get_llm_client() -> 'EnterpriseMistralClient':
    """
    Initializes and returns the singleton LLM client instance.
    """
    global llm_client
    if llm_client is None:
        logger.info("Creating a new singleton instance of the LLM client.")
        llm_client = EnterpriseMistralClient(
            api_key=settings.MISTRAL_API_KEY,
            model_name=settings.MISTRAL_MODEL_NAME
        )
    return llm_client

# Example usage for direct testing of this module
if __name__ == "__main__":
    import asyncio

    async def main():
        print("--- Testing EnterpriseMistralClient ---")
        try:
            client = get_llm_client()

            test_messages = [
                {"role": "system", "content": "You are a helpful assistant that replies succinctly."},
                {"role": "user", "content": "What is the capital of France?"}
            ]

            response_content = await client.invoke_chat_completion(test_messages)

            if response_content:
                print("\n--- Standard Chat Response ---")
                print(response_content)
            else:
                print("Failed to get a standard response.")

        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")

    asyncio.run(main())
