"""
This file is used for RLFT on a Policy (LLM)

Open-source LLMs:
1. [Deepseek-Coder-7b-Instruct-v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
2. [CodeLlama-7b-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

import os
import torch
import textwrap
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from openai import OpenAI
from constants import LORA_CONFIG, RLFT_TRAIN_CONFIG
from src.logger import Logger
from typing import Any, Dict, Optional, List
import time

CONSTANT_PROMPT = textwrap.dedent("""You are a Verilog assistant.  
    Return exactly two blocks in this order:

    <reason>Describe the coding steps in less than 100 words, no commentary outside this tag.</reason>

    ```verilog
    [final Verilog solution only]
    ```

    TASK:
    {prompt}

    Rules:
    1. Output exactly ONE <reason> block and ONE ```verilog block.
    2. Do not repeat modules or add extra text.
    3. The code fence must be ```verilog.
    4. Close the module with endmodule.
    """)


class Lora:
    """
    Applies LoRA adapters
    """

    def __init__(self):
        """
        Load/Store LoRA constants
        """
        self.rank = LORA_CONFIG.get("rank", 16)
        self.alpha = LORA_CONFIG.get("alpha", 32)
        self.dropout = LORA_CONFIG.get("dropout", 0.1)
        self.bias = LORA_CONFIG.get("bias", "none")
        self.target_modules = LORA_CONFIG.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.task_type = LORA_CONFIG.get("task_type", TaskType.CAUSAL_LM)
        self.log = Logger("lora").get_logger()

    def apply(self, model):
        """
        Apply LoRA adapters to the LLM in memory (after loading)
        """
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=self.task_type,
        )

        peft_model = get_peft_model(model, lora_config)
        trainable_params = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in peft_model.parameters())

        self.log.info(
            f"LoRA applied: {trainable_params:,} trainable params out of {total_params:,} total"
        )
        self.log.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        return peft_model


class Policy:
    """
    Responsible to load LLM + LoRA adapters (if required)
    """

    def __init__(
        self,
        name: str,
        unique_id: str,
        grad_check: bool = False,
        apply_lora: bool = True,
        device: Optional[Any] = None,
    ):
        """
        Initialize model name, unique_id and other constants
        """
        self.model = None
        self.tokenizer = None
        self.name = name
        self.unique_id = unique_id
        self.precision = RLFT_TRAIN_CONFIG.get("precision", torch.float16)
        self.apply_lora: bool = RLFT_TRAIN_CONFIG.get("apply_lora", apply_lora)
        self.system_prompt = RLFT_TRAIN_CONFIG.get(
            "system_prompt",
            "You are a Verilog Expert.",
        )
        self.log = Logger(f"policy-{name}").get_logger()
        self.grad_check = grad_check
        self.device = device
        self._select_device()  # Select hardware to load policy

    def _select_device(self):
        """
        Detects if GPU is present in hardware
        Selects it, if found.
        Else, chooses CPU
        """
        if torch.cuda.is_available() and self.device is None:
            self.device = torch.device("cuda")
            self.log.info(f"NVIDIA GPU detected: {torch.cuda.get_device_name()}")
        elif (
            torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
            and self.device is None
        ):
            self.device = torch.device("mps")
            self.log.info("Apple Silicon GPU detected")
        else:
            self.device = torch.device("cpu")
            self.log.info("No GPU detected, using CPU")

    def load(self):
        """
        Load model to CPU or GPU if found
        """
        try:
            # Load HuggingFace API token from environment file
            # Required for accessing gated models like CodeLlama
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError(
                    "'HUGGINGFACE_API_KEY' not found in environment variables"
                )

            # Hugginface Login step
            login(token=api_key)
            self.log.info("Login successful for HUGGING_FACE")

            # Load tokenizer - converts text to tokens the model understands
            self.log.info(f"Model chosen for HUGGING_FACE: {self.unique_id}")
            self.log.info(f"Loading tokenizer from {self.unique_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.unique_id, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load the actual model weights
            # Initialize model with memory-efficient settings
            self.log.info(f"Loading model from HF for: {self.unique_id}...")
            self.log.info(
                "This may take several minutes on first run as it is downloading"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.unique_id,
                torch_dtype=self.precision,
                low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
                trust_remote_code=True,
            ).to(self.device)
            self.log.info("Model loaded successfully!")

            # Apply LoRA if configured
            if self.apply_lora:
                if self.grad_check:
                    # Enable gradient checkpointing BEFORE applying LoRA
                    # Note: Only for training model, not reference
                    self.model.gradient_checkpointing_enable()
                    # IMPORTANT: Enable input gradients for PEFT + gradient checkpointing
                    self.model.enable_input_require_grads()
                    self.log.info(
                        "Gradient checkpointing enabled for memory efficiency"
                    )
                self.log.info("Applying LoRA adapters...")
                lora = Lora()
                self.model = lora.apply(self.model)
                self.log.info("LoRA adapters applied successfully!")
        except ValueError as v_err:
            self.log.error(f"Huggingface API KEY not found in ENV")
            raise ValueError(v_err)
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.unique_id} | Error: {err}"
            )
            raise Exception(f"Exception caught | Reason: {err}")

    def generate(self, prompts: List[str], **kwargs):
        """
        Generate response from LLM using prompt
        Also, generate responses in batches
        """
        # Check time taken to generate
        start_time = time.time()

        # Load constants
        padding = RLFT_TRAIN_CONFIG.get("padding", True)
        truncation = RLFT_TRAIN_CONFIG.get("truncation", True)
        sample_size = RLFT_TRAIN_CONFIG.get("sample_size", 4)
        max_tokens = RLFT_TRAIN_CONFIG.get("max_tokens", 512)
        temperature = RLFT_TRAIN_CONFIG.get("temperature", 0.4)
        top_p = RLFT_TRAIN_CONFIG.get("top_p", 0.9)

        try:
            prompts = [CONSTANT_PROMPT.format(prompt=prompt) for prompt in prompts]
            # Tokenize inputs
            tokenize_start = time.time()
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=padding,
                truncation=truncation,
            ).to(self.device)
            tokenize_time = time.time() - tokenize_start

            # Check time taken to generate
            generation_start = time.time()
            with torch.no_grad():
                for idx, prompt in enumerate(prompts):
                    self.log.info(f"Generating for question {idx}: {prompt}")
                # Generate configuration data
                gen_config = GenerationConfig(
                    max_new_tokens=max_tokens - inputs.input_ids.shape[1],
                    temperature=temperature,
                    do_sample=True if sample_size > 0 else False,
                    top_p=top_p,
                    num_return_sequences=sample_size,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    **kwargs,
                )
            # End of generation
            generation_time = time.time() - generation_start

            # Decode responses
            decode_start = time.time()
            generated_texts = self.tokenizer.batch_decode(
                outputs.sequences[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            decode_time = time.time() - decode_start

            total_time = time.time() - start_time

            # Log timing info
            self.log.info(
                f"Generation complete in {total_time:.2f}s "
                f"(tokenize: {tokenize_time:.3f}s, generate: {generation_time:.2f}s, decode: {decode_time:.3f}s)"
            )

            # Calculate tokens per second
            total_tokens = outputs.sequences.numel() - inputs.input_ids.numel()
            tokens_per_second = (
                total_tokens / generation_time if generation_time > 0 else 0
            )
            self.log.info(
                f"Tokens generated: {total_tokens}, Speed: {tokens_per_second:.1f} tokens/s"
            )

            # If you have multiple prompts with different lengths
            if padding:
                # Each prompt might have different length
                prompt_lengths_per_prompt = (
                    inputs.input_ids != self.tokenizer.pad_token_id
                ).sum(dim=1)
                # Repeat for each sample
                prompt_lengths = prompt_lengths_per_prompt.repeat_interleave(
                    sample_size
                )
            else:
                # All same length
                num_sequences = outputs.sequences.shape[0]
                prompt_lengths = torch.full((num_sequences,), inputs.input_ids.shape[1])

            self.log.info(f"Requested samples: {sample_size}")
            self.log.info(f"Actual sequences generated: {outputs.sequences.shape[0]}")
            self.log.info(
                f"Expected: {len(prompts)} prompts x {sample_size} samples = {len(prompts) * sample_size}"
            )

            return {
                "texts": generated_texts,
                "sequences": outputs.sequences,
                "scores": outputs.scores,
                "prompts_token_length": prompt_lengths,
                "timing": {
                    "total_time": total_time,
                    "tokenize_time": tokenize_time,
                    "generation_time": generation_time,
                    "decode_time": decode_time,
                    "tokens_per_second": tokens_per_second,
                },
            }
        except Exception as err:
            self.log.error(
                f"Error while generating response with model: {self.unique_id} | Error: {err}"
            )
            raise Exception(f"Exception caught | Reason: {err}")


class OpenAIAPIClient:
    """
    SDK client for OpenAI LLM
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Parameters:
        -----------
        model:  str
            Name of the model
            Default: OpenAI GPT 4o

        Returns:
        --------
        None
        """
        self.log = Logger("gpt_api_client").get_logger()
        self.model = model
        self.client = None

    def connect(self) -> bool:
        """
        Establish connection with OpenAI API

        Returns:
        --------
        Bool value to confirm connection
        """
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            self.client = OpenAI(api_key=api_key)
            self.log.info(f"Model chosen for OpenAI: {self.model}")
            return True
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.model} | Error: {err}"
            )
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are an expert programming and reasoning evaluator.",
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate answer via OpenAI API

        Parameters:
        -----------
        prompt:         str
            Natural language description of the Verilog module to generate
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate

        Returns:
        --------
        Dict of all responses

        Raises:
        -------
        openai.OpenAIError
            If API call fails (rate limits, invalid API key, etc.)
        """
        self.log.info("Running chat")

        start_time = time.time()

        # Make API call to OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    # System message instructs the model to behave as a Verilog generator
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,  # The actual Verilog generation request
                },
            ],
            temperature=temperature,  # Control randomness
            max_tokens=max_tokens,  # Limit response length
            n=n_samples,  # Generate multiple samples at once
        )

        end_time = time.time()
        generation_time = end_time - start_time
        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )

        self.log.info("Successfully ran chat")

        # Extract token usage information from the response
        usage = response.usage

        # Calculate tokens per sample (OpenAI gives total, so divide by n_samples)
        # Note: Input tokens are the same for all samples
        tokens_per_sample = []
        if n_samples > 0:
            avg_completion_tokens = usage.completion_tokens / n_samples
            tokens_per_sample = [avg_completion_tokens] * n_samples

        # Calculate tokens per second metrics
        tokens_per_second = (
            usage.completion_tokens / generation_time if generation_time > 0 else 0
        )
        total_tokens_per_second = (
            usage.total_tokens / generation_time if generation_time > 0 else 0
        )

        # Format response to match expected structure for evaluation
        # This ensures consistency across different LLM providers
        response_data = {
            "question": prompt,  # Original prompt for reference
            # Extract generated text from each completion choice
            "outputs": [choice.message.content for choice in response.choices],
            "config": {  # Configuration used for generation
                "model": self.model,
                "system_instruction": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
            "time": generation_time,
            # Token usage information
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "tokens_per_sample": tokens_per_sample,
            "avg_tokens_per_sample": usage.completion_tokens / n_samples
            if n_samples > 0
            else 0,
            "output_tokens_per_second": tokens_per_second,
            "total_tokens_per_second": total_tokens_per_second,
        }

        return response_data
