from openai import OpenAI
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
import aiohttp
from tqdm import tqdm
import random
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

def openai_query(client, model, prompt, sampling_params):
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
            responses = [x.message.content for x in completion.choices]
        # responses.append(response)
    else:
        completion = client.responses.create(
            model="o3-mini",
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
        # print(completion)
        cost = compute_openai_api_cost_reasoning(completion, model)
        responses = []
        responses.append(completion.output_text)
    
    return responses, cost

class OpenaiClient:
    def __init__(self, model="gpt-4o"):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        # api_key_list = [
        #     "YOUR_OPENAI_API_KEY_1_HERE",
        #     "YOUR_OPENAI_API_KEY_2_HERE"
        # ]
        # self.client_list = [OpenAI(api_key=key) for key in api_key_list]
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.total_cost = 0.0
        self.sampling_params = SamplingParams(temperature=0.0, n=1, max_tokens=100)
    
    # @property
    # def client(self):
    #     return random.choice(self.client_list)
    
    def _get_token_param_name(self):
        """Get the appropriate parameter name for token limits based on the model"""
        models_with_max_completion_tokens = [
            "gpt-5", "gpt-5-mini", "gpt-5o", "gpt-5o-mini","gpt-5-nano", 
            "gpt-4.5", "gpt-4.5-mini"  # Future models that might use this parameter
        ]
        return "max_completion_tokens" if self.model in models_with_max_completion_tokens else "max_tokens"
    
    def query(self, prompt, temperature=0.0, n=1, max_tokens=100):
        """Query the OpenAI API with the given prompt"""
        sampling_params = SamplingParams(temperature=temperature, n=n, max_tokens=max_tokens)
        responses, cost = openai_query(self.client, self.model, prompt, sampling_params)
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

    def batch_query_multiprocessing(self, prompts, temperature=0.0, max_tokens=100, max_workers=None, chunk_size=10, show_progress=True):
        """
        Process multiple queries concurrently using multiprocessing.
        
        Args:
            prompts (list): List of prompt strings to process
            temperature (float): Sampling temperature for all queries
            max_tokens (int): Maximum tokens for all queries
            max_workers (int): Maximum number of worker processes (defaults to CPU count)
            chunk_size (int): Number of prompts to process in each worker batch
            show_progress (bool): Whether to show progress bar (default: True)
            
        Returns:
            tuple: (responses, total_cost)
                - responses: List of responses in the same order as prompts
                - total_cost: Total cost of all queries
        """
        if not prompts:
            return [], 0.0
            
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(prompts))
        
        # Initialize progress bar
        if show_progress:
            pbar = tqdm(total=len(prompts), desc="Processing API calls (Multiprocessing)", unit="query")
        
        # Create a partial function with the client parameters
        query_func = partial(
            self._single_query_worker,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            pbar=pbar if show_progress else None
        )
        
        responses = [None] * len(prompts)
        total_cost = 0.0
        
        # Process prompts in chunks to avoid overwhelming the API
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
    
    def _single_query_worker(self, prompt, index, model, temperature, max_tokens, pbar=None):
        """
        Worker function for multiprocessing. Creates a new client instance for each process.
        
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
            # Create a new client instance for this process
            # Note: API key is not passed here as it should be set as environment variable
            client = OpenAI(api_key=self.api_key)
            
            # Create sampling params
            sampling_params = SamplingParams(
                temperature=temperature,
                n=1,
                max_tokens=max_tokens
            )
            
            # Process the query
            responses, cost = openai_query(client, model, prompt, sampling_params)
            
            # Extract the response
            response = responses[0] if isinstance(responses[0], str) else responses[0][0]
            
            if pbar:
                pbar.update(1)
            return index, response, cost
            
        except Exception as e:
            print(f"Worker error for prompt {index}: {e}")
            if pbar:
                pbar.update(1)
            return None

    async def batch_query_async(self, prompts, temperature=0.0, max_tokens=100, max_concurrent=10, show_progress=True):
        """
        Process multiple queries concurrently using asyncio (recommended for API calls).
        
        Args:
            prompts (list): List of prompt strings to process
            temperature (float): Sampling temperature for all queries
            max_tokens (int): Maximum tokens for all queries
            max_concurrent (int): Maximum concurrent API calls (default: 10)
            show_progress (bool): Whether to show progress bar (default: True)
            
        Returns:
            tuple: (responses, total_cost)
                - responses: List of responses in the same order as prompts
                - total_cost: Total cost of all queries
        """
        if not prompts:
            return [], 0.0
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize progress bar
        if show_progress:
            pbar = tqdm(total=len(prompts), desc="Processing API calls", unit="query")
        
        # Create tasks for all prompts
        tasks = []
        for i, prompt in enumerate(prompts):
            task = self._async_query_worker(prompt, i, temperature, max_tokens, semaphore, pbar if show_progress else None)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Close progress bar
        if show_progress:
            pbar.close()
        
        # Process results
        responses = [None] * len(prompts)
        total_cost = 0.0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                responses[i] = f"Error: {str(result)}"
                print(f"Error processing query {i}: {result}")
            elif result is not None:
                index, response, cost = result
                responses[index] = response
                total_cost += cost
        
        self.total_cost += total_cost
        return responses, total_cost
    
    async def _async_query_worker(self, prompt, index, temperature, max_tokens, semaphore, pbar=None):
        """
        Async worker function for concurrent API calls.
        
        Args:
            prompt (str): The prompt to process
            index (int): Index of the prompt in the original list
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens
            semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
            pbar (tqdm): Progress bar instance (optional)
            
        Returns:
            tuple: (index, response, cost) or None if failed
        """
        async with semaphore:  # Limit concurrent requests
            try:
                # Create a new client instance for this request
                client = OpenAI(api_key=self.api_key)
                
                # Create sampling params
                sampling_params = SamplingParams(
                    temperature=temperature,
                    n=1,
                    max_tokens=max_tokens
                )
                
                # Process the query
                responses, cost = openai_query(client, self.model, prompt, sampling_params)
                
                # Extract the response
                response = responses[0] if isinstance(responses[0], str) else responses[0][0]
                
                if pbar:
                    pbar.update(1)
                return index, response, cost
                
            except Exception as e:
                print(f"Async worker error for prompt {index}: {e}")
                if pbar:
                    pbar.update(1)
                return None
    
    def batch_query_async_sync(self, prompts, temperature=0.0, max_tokens=100, max_concurrent=10, show_progress=True):
        """
        Synchronous wrapper for async batch query method.
        Useful when you want to use async functionality in a sync context.
        
        Args:
            prompts (list): List of prompt strings to process
            temperature (float): Sampling temperature for all queries
            max_tokens (int): Maximum tokens for all queries
            max_concurrent (int): Maximum concurrent API calls
            show_progress (bool): Whether to show progress bar (default: True)
            
        Returns:
            tuple: (responses, total_cost)
        """
        return asyncio.run(self.batch_query_async(prompts, temperature, max_tokens, max_concurrent, show_progress))

    def batch_query_threading(self, prompts, temperature=0.0, max_tokens=100, max_workers=None, show_progress=True):
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
            pbar=pbar if show_progress else None
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
    
    def _thread_query_worker(self, prompt, index, model, temperature, max_tokens, pbar=None):
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
            # Create a new client instance for this thread
            client = OpenAI(api_key=self.api_key)
            
            # Create sampling params
            sampling_params = SamplingParams(
                temperature=temperature,
                n=1,
                max_tokens=max_tokens
            )
            
            # Process the query
            responses, cost = openai_query(client, model, prompt, sampling_params)
            
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

if __name__ == "__main__":
    # Example usage of the OpenaiClient class
    api_key = "YOUR_OPENAI_API_KEY_HERE"  # Replace with your actual API key
    
    # Create client instance
    model = OpenaiClient(api_key=api_key, model="gpt-5")
    
    # Query the model
    # prompt = "Hi, how are you doing today?"
    # response = model.query(prompt)
    # print(f"Response: {response}")
    
    # # Get the cost
    # cost = model.get_cost()
    # print(f"Cost: ${cost:.6f}")
    
    # # Example with custom parameters
    # response2 = model.query("What is 2+2?", temperature=0.1, max_tokens=50)
    # print(f"Response 2: {response2}")
    
    # Example of multiprocessing batch queries
    print("\n--- Multiprocessing Batch Queries ---")
    batch_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "What is 15 * 23?",
        "Name three programming languages.",
        "What is the largest planet in our solar system?"
    ]
    
    # Set your API key as environment variable for multiprocessing
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Process batch queries concurrently with progress bar
    responses, batch_cost = model.batch_query_multiprocessing(
        batch_prompts, 
        temperature=0.0, 
        max_tokens=100,
        max_workers=3,  # Limit to 3 concurrent processes
        show_progress=True  # Show progress bar
    )
    
    print(f"Batch responses:")
    for i, (prompt, response) in enumerate(zip(batch_prompts, responses)):
        print(f"{i+1}. Q: {prompt}")
        print(f"   A: {response}")
        print()
    
    print(f"Batch cost: ${batch_cost:.6f}")
    
    # Example of threading batch queries (often better than multiprocessing for API calls)
    print("\n--- Threading Batch Queries ---")
    responses_threading, batch_cost_threading = model.batch_query_threading(
        batch_prompts, 
        temperature=1.0, 
        max_tokens=100,
        max_workers=5,  # Can handle more concurrent threads
        show_progress=True  # Show progress bar
    )
    
    print(f"Threading batch responses:")
    for i, (prompt, response) in enumerate(zip(batch_prompts, responses_threading)):
        print(f"{i+1}. Q: {prompt}")
        print(f"   A: {response}")
        print()
    
    print(f"Threading batch cost: ${batch_cost_threading:.6f}")
    
    # # Example of async batch queries (best for API calls)
    # print("\n--- Async Batch Queries ---")
    # responses_async, batch_cost_async = model.batch_query_async_sync(
    #     batch_prompts, 
    #     temperature=0.0, 
    #     max_tokens=100,
    #     max_concurrent=8,  # Can handle many concurrent requests
    #     show_progress=True  # Show progress bar
    # )
    
    # print(f"Async batch responses:")
    # for i, (prompt, response) in enumerate(zip(batch_prompts, responses_async)):
    #     print(f"{i+1}. Q: {prompt}")
    #     print(f"   A: {response}")
    #     print()
    
    # print(f"Async batch cost: ${batch_cost_async:.6f}")
    
    # # Example of disabling progress bars
    # print("\n--- Async Batch Queries (No Progress Bar) ---")
    # responses_async_no_progress, batch_cost_async_no_progress = model.batch_query_async_sync(
    #     batch_prompts, 
    #     temperature=0.0, 
    #     max_tokens=100,
    #     max_concurrent=8,
    #     show_progress=False  # Disable progress bar
    # )
    
    # print(f"Async batch responses (no progress bar):")
    # for i, (prompt, response) in enumerate(zip(batch_prompts, responses_async_no_progress)):
    #     print(f"{i+1}. Q: {prompt}")
    #     print(f"   A: {response}")
    #     print()
    
    # print(f"Async batch cost (no progress bar): ${batch_cost_async_no_progress:.6f}")
    
    # Get total cost
    total_cost = model.get_cost()
    print(f"Total cost: ${total_cost:.6f}")