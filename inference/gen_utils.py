from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
import argparse
from dataclasses import replace
from config import config
from transformers import AutoTokenizer
from functools import partial
from vllm.sampling_params import BeamSearchParams

def update_config_from_args():
    """Update configuration based on command line arguments."""
    parser = argparse.ArgumentParser(description='Generate responses with configurable parameters')
    
    # Model generation parameters
    parser.add_argument('--temperature', type=float, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, help='Nucleus sampling parameter')
    parser.add_argument('--max_length', type=int, help='Maximum length of generated response')
    parser.add_argument('--num_samples', type=int, help='Number of samples per question')
    parser.add_argument('--base_model', type=str, help='Base model to use')
    
    # Question and prompt type parameters
    parser.add_argument('--question_type', type=str, default='general', help='Type of questions')
    parser.add_argument('--prompt_type', type=str, help='Type of prompt template to use')
    
    args = parser.parse_args()
    
    # Update model config with provided arguments
    model_updates = {}
    if args.temperature is not None:
        model_updates['temperature'] = args.temperature
    if args.top_p is not None:
        model_updates['top_p'] = args.top_p
    if args.max_length is not None:
        model_updates['max_length'] = args.max_length
    if args.num_samples is not None:
        model_updates['num_samples_per_question'] = args.num_samples
    if args.base_model is not None:
        model_updates['base_model'] = args.base_model
        
    if model_updates:
        config.model = replace(config.model, **model_updates)
    
    return args

def postprocess_generated_output(output, use_beam_search=False):
    def _sum_logprobs(logprobs):
        sum_ = 0.0
        for item in logprobs:
            assert len(item) == 1
            for k in item:
                sum_ += item[k].logprob
        return sum_
    
    # processed = {"response": [], "cum_logprobs": [], "total_length": []}
    processed = []
    if use_beam_search:
        for item in output.sequences:
            item_res = {}
            item_res["response"] = item.text
            # if hasattr(item, "cum_logprob"):
            item_res["cum_logprobs"] = item.cum_logprob
            item_res["total_length"] = len(item.tokens)
            processed.append(item_res)
    else:
        for item in output.outputs:
            item_res = {}
            item_res["response"]  = item.text
            if hasattr(item, "logprobs") and item.logprobs is not None:
                item_res["cum_logprobs"]  = _sum_logprobs(item.logprobs)
                item_res["total_length"]  = len(item.logprobs)
            processed.append(item_res)
    return processed
        



class ResponseGenerator:
    def __init__(self, model_name: str = None, tokenizer_name: str = None, gpu_memory_util: float = 0.75, lora_dir: str = None):
        """Initialize the response generator with a specific model."""
        self.model_name = model_name
        self.lora_dir = lora_dir
        print("ResponseGenerator: model_name: ", self.model_name)
        lora_kwargs = dict(enable_lora=True, max_lora_rank=32) if lora_dir else {}
        # or config.model.base_model
        self.llm = LLM(model=self.model_name, tokenizer=tokenizer_name, dtype=torch.float16, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=gpu_memory_util, enforce_eager=True, **lora_kwargs)
        self.prompt_config = config.prompt
        self.model_config = config.model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else self.model_name)
    
    def generate_responses(self, questions: List[Dict[str, Any]], 
                         question_type: str = "general",
                         prompt_type: Optional[str] = None,
                         temperature: Optional[float] = None,
                         top_p: Optional[float] = None,
                         max_length: Optional[int] = None,
                         num_samples: Optional[int] = None,
                         beam_search=False,
                         logprob=True) -> List[Dict[str, Any]]:
        """Generate multiple responses for each question.
        
        Args:
            questions: List of question dictionaries
            question_type: Type of questions (e.g., "math", "general")
                         Used to determine default generation parameters if not specified
            prompt_type: Type of prompt to use (if None, uses default for question_type)
            temperature: Sampling temperature (overrides type-specific parameter)
            top_p: Nucleus sampling parameter (overrides type-specific parameter)
            max_length: Maximum length of generated response (overrides type-specific parameter)
            num_samples: Number of samples to generate per question (overrides type-specific parameter)
        """
        if beam_search:
            self.llm.llm_engine.model_config.max_logprobs = 2 * num_samples
        all_responses = []
        
        # Get type-specific parameters as defaults
        default_params = self._get_generation_params(question_type)
        
        # Override defaults with provided parameters
        gen_params = {
            "temperature": temperature if temperature is not None else default_params["temperature"],
            "top_p": top_p if top_p is not None else default_params["top_p"],
            "max_length": max_length if max_length is not None else default_params["max_length"],
            "num_samples": num_samples if num_samples is not None else default_params["num_samples"]
        }

        all_questions = []
        all_prompts = []
        
        for question in tqdm(questions, desc="Generating responses"):
            # Get question-specific parameters (question-level overrides have highest priority)
            question_params = {
                "num_samples": question.get("num_samples", gen_params["num_samples"]),
                "temperature": question.get("temperature", gen_params["temperature"]),
                "max_length": question.get("max_length", gen_params["max_length"]),
                "top_p": question.get("top_p", gen_params["top_p"])
            }
            
            # Format prompt using appropriate template
            prompt = self._format_prompt(
                question["question"], 
                prompt_type=prompt_type,
                template=question.get("prompt_template", None)
            )

            if not beam_search:
                all_questions.extend([question] * question_params["num_samples"])
                all_prompts.extend([prompt] * question_params["num_samples"])
            else:
                all_questions.append(question)
                all_prompts.append(prompt)
                # For beam search, we only need one prompt per question
        
        if beam_search:
            sampling_params = BeamSearchParams(
                beam_width=num_samples,
                max_tokens=question_params["max_length"],
                stop=["Question:", "</s>", self.tokenizer.eos_token] if prompt_type == "simplerl_base" else None
            )
        else:
            # Generate multiple samples for each question
            sampling_params = SamplingParams(
                temperature=question_params["temperature"],
                top_p=question_params["top_p"],
                max_tokens=question_params["max_length"],
                stop=["Question:", "</s>", self.tokenizer.eos_token] if prompt_type == "simplerl_base" else None
            )

        # print example prompt
        print("========== Example prompt ==========")
        print(all_prompts[0])
        print("========== ============== ==========")
        lora_request = LoRARequest("adapter", 1, self.lora_dir) if self.lora_dir else None
        if not beam_search:
            responses = self.llm.generate(
                prompts=all_prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
        else:
            print("beam search ...")
            responses = []
            bsz = 20
            for i in range(0, len(all_prompts), bsz):
                batch = all_prompts[i:i+bsz]
                responses_tmp = self.llm.beam_search(
                    batch,
                    sampling_params,
                    lora_request=lora_request,
                )
                responses.extend(responses_tmp)
                print("finished: ", len(responses))
        
        processed_responses = map(partial(postprocess_generated_output, use_beam_search=beam_search), responses)
        
        for response_list, question in zip(processed_responses, all_questions):
            for response in response_list:
                # Store responses with metadata
                response_data = {
                    "question_id": question["id"],
                    "question": question["question"],
                    "response": response["response"],
                    "metadata": {
                        "temperature": question_params["temperature"],
                        "top_p": question_params["top_p"],
                        "question_type": question_type,
                        "prompt_type": prompt_type,
                        "prompt_template": prompt
                    }
                }
                # Include ground truth if provided
                if "ground_truth" in question:
                    response_data["ground_truth"] = question["ground_truth"]
                if "cum_logprobs" in response:
                    response_data["cum_logprobs"] = response["cum_logprobs"]
                if "total_length" in response:
                    response_data["total_length"] = response["total_length"]

                    
                all_responses.append(response_data)
        
        return all_responses
    
    def _format_prompt(self, question: str, prompt_type: str = "general", 
                      template: Optional[str] = None) -> str:
        """Format the question into a proper prompt using the appropriate template.
        
        Args:
            question: The question text
            prompt_type: Type of prompt to use (e.g., "math", "general")
            template: Optional custom template to use for this specific question
        """
        if template is not None:
            # Use provided custom template
            prompt_template = template
        elif prompt_type in self.prompt_config.custom_templates:
            # Use prompt type specific template
            prompt_template = self.prompt_config.custom_templates[prompt_type]
        elif prompt_type == "math":
            # Use math-specific template
            prompt_template = self.prompt_config.math_template
        else:
            # Use default template
            prompt_template = self.prompt_config.default_template
        prompt = prompt_template.format(question=question)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        return prompt
    
    def _get_generation_params(self, question_type: str = "general") -> Dict[str, Any]:
        """Get generation parameters for the specified question type."""
        if question_type in self.model_config.type_specific_params:
            params = self.model_config.type_specific_params[question_type].copy()
        else:
            params = {
                "temperature": self.model_config.temperature,
                "top_p": self.model_config.top_p,
                "max_length": self.model_config.max_length,
                "num_samples": self.model_config.num_samples_per_question
            }
        return params
    
    def save_responses(self, responses: List[Dict[str, Any]], output_file: str):
        """Save generated responses to a file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(responses, f, indent=2)

def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """Load questions from a JSON file."""
    with open(questions_file, 'r') as f:
        return json.load(f)

