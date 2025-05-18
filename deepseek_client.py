import re
import time
import logging
import json
from typing import Dict, List, Optional, Any
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logger = logging.getLogger(__name__)

class DeepSeekClient:
    def __init__(
        self,
        model_name: str = "deepseek-r1:7b",
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 10.0
    ):
        """
        Initialize the DeepSeek client.
        
        Args:
            model_name: The name of the DeepSeek model in Ollama
            temperature: Controls randomness in model responses (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_base_delay: Base delay for retry exponential backoff
            retry_max_delay: Maximum delay between retries
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        
        # Validate the model is available
        self._validate_model()
    
    def _validate_model(self) -> None:
        """
        Validate that the specified model is available in Ollama.
        
        Raises:
            ValueError: If the model is not available
        """
        try:
            models_response = ollama.list()
            
            if hasattr(models_response, 'models'):
                models = models_response.models
            else:
                # Fall back to treating it as a dict if needed
                models = models_response.get('models', [])
                
            # Extract model names - handle both object and dict formats
            model_names = []
            for model in models:
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                elif isinstance(model, dict) and 'name' in model:
                    model_names.append(model.get('name'))
                elif isinstance(model, dict) and 'model' in model:
                    model_names.append(model.get('model'))
            
            if self.model_name not in model_names:
                logger.warning(
                    f"Model '{self.model_name}' not found in Ollama. "
                    f"Available models: {', '.join(model_names)}"
                )
                logger.info(f"You can pull it with: ollama pull {self.model_name}")
            else:
                logger.info(f"Model '{self.model_name}' is available in Ollama")
                    
        except Exception as e:
            logger.error(f"Error validating model: {str(e)}")
            logger.warning("Make sure Ollama is running and accessible")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        raw_response: bool = False,
        stream: bool = False
    ) -> str:
        """
        Generate a response from the DeepSeek model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system instruction to guide the model
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            timeout: Request timeout in seconds
            raw_response: If True, return the raw response without cleaning
            stream: If True, return a generator for streaming responses
            
        Returns:
            The model's response as a string
            
        Raises:
            ValueError: For invalid inputs
            RuntimeError: For model execution errors
            TimeoutError: If the request times out
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Build options
        options = {
            "temperature": temperature if temperature is not None else self.temperature,
        }
        
        if max_tokens is not None or self.max_tokens is not None:
            options["num_predict"] = max_tokens if max_tokens is not None else self.max_tokens
        
        # Set timeout
        timeout_value = timeout if timeout is not None else self.timeout
        
        try:
            start_time = time.time()
            
            if stream:
                return self._stream_response(messages, options)
            
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options=options
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Request completed in {elapsed_time:.2f} seconds")
            
            if not response or "message" not in response:
                raise RuntimeError("Received invalid response from model")
            
            content = response["message"]["content"]
            
            if raw_response:
                return content
            
            return self._clean_response(content)
            
        except ollama.ResponseError as e:
            logger.error(f"Ollama response error: {str(e)}")
            raise RuntimeError(f"Model execution error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _stream_response(self, messages: List[Dict[str, str]], options: Dict[str, Any]):
        """
        Stream responses from the model.
        
        Args:
            messages: List of message objects
            options: Generation options
            
        Yields:
            Chunks of the response as they're generated
        """
        try:
            for chunk in ollama.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                stream=True
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            raise RuntimeError(f"Streaming error: {str(e)}")
    
    def _clean_response(self, text: str) -> str:
        """
        Clean LLM response by removing thinking artifacts and other unwanted content.
        
        Args:
            text: Raw model response
            
        Returns:
            Cleaned response text
        """
        if not text:
            return ""
        
        # Remove thinking sections
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove other potential artifacts
        text = re.sub(r'<userStyle>.*?</userStyle>', '', text, flags=re.DOTALL)
        text = re.sub(r'<internal>.*?</internal>', '', text, flags=re.DOTALL)
        
        # Trim whitespace
        return text.strip()
    
    def extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from model response.
        
        Args:
            text: Response text from the model
            
        Returns:
            Extracted JSON as a dictionary
            
        Raises:
            ValueError: If JSON cannot be extracted
        """
        # First attempt: Look for JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Second attempt: Look for JSON with curly braces
        json_match = re.search(r'(\{[\s\S]*\})', text, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Third attempt: Try to parse the entire string
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        raise ValueError("Could not extract valid JSON from model response")