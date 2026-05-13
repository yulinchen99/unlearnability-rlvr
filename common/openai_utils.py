from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
"""
this file contains the utils for openai api
usage:
from openai_utils import OpenaiClient
model = OpenaiClient(api_key="your_api_key", model="gpt-4o")
prompt = "Hi, how are you doing today?"
response = model.query(prompt)
cost = model.get_cost()
print(cost)
"""

class SamplingParams:
    def __init__(self, temperature=0.0, n=1, max_tokens=100):
        self.temperature = temperature
        self.n = n
        self.max_tokens = max_tokens

    def __repr__(self):
        return f"SamplingParams(temperature={self.temperature}, n={self.n}, max_tokens={self.max_tokens})"


def compute_openai_api_cost(completion, model="gpt-4o"):
    model_cost = {
        "gpt-4o": {"input": 2.5, "output": 10}, 
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},  
        "o3-mini": {"input": 1.1, "output": 4.4},
        "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
        "gpt-5-mini": {"input": 0.25, "output": 2.0},
        "gpt-5": {"input": 1.25, "output": 10.0},
        "gpt-5-nano": {"input": 0.05, "output": 0.4},
        "deepseek-reasoner": {"input": 0.28, "output": 0.42},
        }
    assert model in model_cost
    cost = model_cost[model]["input"] / (1_000_000) * completion.usage.prompt_tokens
    cost += model_cost[model]["output"] / (1_000_000) * completion.usage.completion_tokens
    return cost

def compute_openai_api_cost_reasoning(completion, model="gpt-4o"):
    model_cost = {"gpt-4o": {"input": 2.5, "output": 10}, "gpt-4o-mini": {"input": 0.15, "output": 0.6},  "o3-mini": {"input": 1.1, "output": 4.4}}
    assert model in model_cost
    cost = model_cost[model]["input"] / (1_000_000) * completion.usage.input_tokens
    cost += model_cost[model]["output"] / (1_000_000) * completion.usage.output_tokens
    return cost

def openai_query(client, model, prompt, sampling_params, **kwargs):
    """
    Uses openAI API to query
    """
    # Models that use max_completion_tokens instead of max_tokens
    # This includes newer models like GPT-5 and potentially future models
    models_with_max_completion_tokens = [
        "gpt-5", "gpt-5-mini", "gpt-5o", "gpt-5o-mini", "gpt-5-nano",
        "gpt-4.5", "gpt-4.5-mini"  # Future models that might use this parameter
    ]
    
    if model not in ["o3-mini"]:
        # Prepare parameters based on model
        api_params = {
            "model": model,
            "n": sampling_params.n,
            "temperature": sampling_params.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Use appropriate parameter name for token limit
        if model in models_with_max_completion_tokens:
            api_params["max_completion_tokens"] = sampling_params.max_tokens
        else:
            api_params["max_tokens"] = sampling_params.max_tokens
        
        completion = client.chat.completions.create(**api_params)
        # print(completion)
        cost = compute_openai_api_cost(completion, model)
        responses = []

        if sampling_params.n == 1:
            responses.append(completion.choices[0].message.content)
        else:
            response = [x.message.content for x in completion.choices]
            responses.append(response)
    else:
        completion = client.responses.create(
            model="o3-mini",
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            **kwargs
        )
        # print(completion)
        cost = compute_openai_api_cost_reasoning(completion, model)
        responses = []
        responses.append(completion.output_text)
    return responses, cost

class OpenaiClient:
    def __init__(self, model="gpt-4o", api_key=None):
        if "deepseek" in model:
            self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise RuntimeError("DEEPSEEK_API_KEY not set")
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.total_cost = 0.0
        self.sampling_params = SamplingParams(temperature=0.0, n=1, max_tokens=100)
    
    def _get_token_param_name(self):
        """Get the appropriate parameter name for token limits based on the model"""
        models_with_max_completion_tokens = [
            "gpt-5", "gpt-5-mini", "gpt-5o", "gpt-5o-mini","gpt-5-nano", 
            "gpt-4.5", "gpt-4.5-mini"  # Future models that might use this parameter
        ]
        return "max_completion_tokens" if self.model in models_with_max_completion_tokens else "max_tokens"
    
    def query(self, prompt, temperature=0.0, n=1, max_tokens=100, **kwargs):
        """Query the OpenAI API with the given prompt"""
        self.sampling_params = SamplingParams(temperature=temperature, n=n, max_tokens=max_tokens)
        responses, cost = openai_query(self.client, self.model, prompt, self.sampling_params, **kwargs)
        self.total_cost += cost
        
        # Return single response if n=1, otherwise return list
        if n == 1:
            return responses[0] if isinstance(responses[0], str) else responses[0][0]
        return responses[0] if isinstance(responses[0], list) else responses
    
    def get_cost(self):
        """Get the total cost of all queries made"""
        return self.total_cost
    
    def reset_cost(self):
        """Reset the total cost counter"""
        self.total_cost = 0.0

    def batch_query_threading(self, prompts, temperature=0.0, max_tokens=100, max_workers=None, show_progress=True, **kwargs):
        """
        Process multiple queries concurrently using ThreadPoolExecutor.
        Good for I/O-bound operations like API calls, with less overhead than multiprocessing.
        
        Args:
            prompts (list): List of prompt strings to process
            temperature (float): Sampling temperature for all queries
            max_tokens (int): Maximum tokens for all queries
            max_workers (int): Maximum number of worker threads (defaults to min(32, len(prompts)))
            show_progress (bool): Whether to show progress bar (default: True)
            
        Returns:
            tuple: (responses, total_cost)
                - responses: List of responses in the same order as prompts
                - total_cost: Total cost of all queries
        """
        if not prompts:
            return [], 0.0
            
        if max_workers is None:
            max_workers = min(32, len(prompts))  # ThreadPoolExecutor default max is 32
        
        # Initialize progress bar
        if show_progress:
            pbar = tqdm(total=len(prompts), desc="Processing API calls (Threading)", unit="query")
        
        # Create a partial function with the client parameters
        query_func = partial(
            self._thread_query_worker,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            pbar=pbar if show_progress else None,
            **kwargs
        )
        
        responses = [None] * len(prompts)
        total_cost = 0.0
        
        # Process prompts using thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, prompt in enumerate(prompts):
                future = executor.submit(query_func, prompt, i)
                future_to_index[future] = i
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    if result is not None:
                        index, response, cost = result
                        responses[index] = response
                        total_cost += cost
                except Exception as e:
                    print(f"Error processing query: {e}")
                    # Mark failed queries with error message
                    index = future_to_index[future]
                    responses[index] = f"Error: {str(e)}"
        
        # Close progress bar
        if show_progress:
            pbar.close()
        
        self.total_cost += total_cost
        return responses, total_cost
    
    def _thread_query_worker(self, prompt, index, model, temperature, max_tokens, pbar=None, **kwargs):
        """
        Worker function for threading. Creates a new client instance for each thread.
        
        Args:
            prompt (str): The prompt to process
            index (int): Index of the prompt in the original list
            model (str): Model name
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens
            pbar (tqdm): Progress bar instance (optional)
            
        Returns:
            tuple: (index, response, cost) or None if failed
        """
        try:
            if "deepseek" in model:
                client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
            else:
                client = OpenAI(api_key=self.api_key)
            # Create a new client instance for this thread
            # client = OpenAI(api_key=self.api_key)
            
            # Create sampling params
            sampling_params = SamplingParams(
                temperature=temperature,
                n=1,
                max_tokens=max_tokens
            )
            
            # Process the query
            responses, cost = openai_query(client, model, prompt, sampling_params, **kwargs)
            
            # Extract the response
            response = responses[0] if isinstance(responses[0], str) else responses[0][0]
            
            if pbar:
                pbar.update(1)
            return index, response, cost
            
        except Exception as e:
            print(f"Thread worker error for prompt {index}: {e}")
            if pbar:
                pbar.update(1)
            return None