"""
Main driver for all python scripts

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

import glob
import json
import os
import sys
from dotenv import load_dotenv
from typing import Dict, List
from src.gcp import GoogleStorageClient
from src.logger import Logger
from src.models import (
    ClaudeAPIClient,
    GeminiAPIClient,
    OpenAIAPIClient,
    OpenSourceLLMClient,
)


# TODO: load env
class ENVLoader:
    """
    Loads all the ENV required in the local machine
    """

    def __init__(self):
        """
        Loads when object is instantiated
        """
        self.log = Logger("env_loader").get_logger()
        self._run()  # run when object is instantiated

    def _run(self):
        """
        Private method to load ENV
        """
        try:
            load_dotenv("secrets/models-api.env")
            self.log.info("ENV loaded successfully")
        except Exception as err:
            self.log.critical(f"Fatal Error on ENV loading: {err}")
            self.log.info("Exiting ...")
            sys.exit(-1)


# TODO: run prompts
# TODO: end operation
class RunLLMPrompts:
    """
    Handles operation to run prompts for each model
    Helps in deducing model's accuracy in generating Verilog code with eval metrics
    """

    def __init__(self):
        """
        Loads when object is instantiated
        """
        self.log = Logger("llm_runner").get_logger()
        self.dataset_path = "dataset/testbench/verigen"
        self.models = [
            {"id": "code-llama", "name": "meta-llama/CodeLlama-7b-Instruct-hf"},
            {
                "id": "deepseek-coder",
                "name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            },
            {
                "id": "Verigen-6B-Finetuned",
                "name": "shailja/fine-tuned-codegen-6B-Verilog",
            },
            {"id": "qwen-coder", "name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
        ]
        self.claude_api = ClaudeAPIClient()
        self.gemini_api = GeminiAPIClient()
        self.openai_api = OpenAIAPIClient()
        self.prompts = []  # empty list
        self.answers = {}  # empty dict

    def collect_prompts(self) -> List[Dict]:
        """
        From the dataset, collect all the prompts

        Returns:
        --------
        List of Dict of all prompts and testbench code (from VeriGen Repo)
        """
        # Get all directories in test folder
        # Each directory contains one problem with metadata.json
        dirs = os.listdir(self.dataset_path)
        dirs.sort()
        self.log.info(f"Sorted DIRS from path: {self.dataset_path}")

        # Process each problem directory
        for dir in dirs:
            # Skip macOS metadata file
            if dir in [
                ".DS_Store",
                "prompts-summary.txt",
                "prompts-templates.txt",
                "test_ex.v",
            ]:
                self.log.info(f"Omiting misc DIR: {dir}")
                continue
            else:
                pattern = lambda file: os.path.join(self.dataset_path, dir, file)
                # answer_filepath = (glob.glob(pattern("answer_*"), recursive=True))[0]
                prompt_filepath = glob.glob(pattern("prompt1_*"), recursive=True)[0]
                tb_filepath = glob.glob(pattern("tb_*"), recursive=True)[0]

                # answer = None
                prompt = None
                testbench = None

                # with open(answer_filepath, "r") as fs:
                #     answer = fs.read()
                with open(prompt_filepath, "r") as fs:
                    prompt = fs.read()
                with open(tb_filepath, "r") as fs:
                    testbench = fs.read()

                self.log.info(f"Prompt: {prompt}")
                self.prompts.append(
                    {
                        # "answer": answer,
                        "prompt": prompt,
                        "testbench": testbench,
                    }
                )

        self.log.info(f"Created all prompts!")
        prompts = self.prompts
        return prompts

    def run_all_models(
        self,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
        prompt_size: int = None,
    ) -> List[Dict]:
        """
        Runner method for all LLMs

        Parameters:
        -----------
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate
        prompt_size:    int, default=None
            Size of array of the prompts from the Problem set

        Returns:
        --------
        List of Dict of all answers generated by all models, and all filenames of the output
        """
        exception = ""  # For final log note
        filenames = []  # For filenames to copy to GCP storage

        # First case: Claude API
        res = self.claude_api.connect()
        if res:
            self.answers["claude"] = []
            for data in self.prompts[:prompt_size]:
                response = self.claude_api.generate(
                    prompt=data["prompt"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n_samples=n_samples,
                )
                self.answers["claude"].append(
                    {"response": response, "testbench": data["testbench"]}
                )
            with open("prompts-claude.json", "w") as fs:
                json.dump(self.answers["claude"], fs, indent=4)
            self.log.info(f"Written file for claude")
            filenames.append("prompts-claude.json")
        else:
            self.log.error(f"Claude connection unavailable; Moving to next case...")
            exception = "Except Claude"

        # Second case: Gemini API
        # res = self.gemini_api.connect()
        # if res:
        #     self.answers["gemini"] = []
        #     for data in self.prompts[:prompt_size]:
        #         response = self.gemini_api.generate(
        #             prompt=data["prompt"],
        #             temperature=temperature,
        #             max_tokens=max_tokens,
        #             n_samples=n_samples,
        #         )
        #         self.answers["gemini"].append(
        #             {"response": response, "testbench": data["testbench"]}
        #         )
        #     with open("prompts-gemini.json", "w") as fs:
        #         json.dump(self.answers["gemini"], fs, indent=4)
        #     self.log.info(f"Written file for gemini")
        #     filenames.append("prompts-gemini.json")
        # else:
        #     self.log.error(f"Gemini connection unavailable; Moving to next case...")
        #     if len(exception) == 0:
        #         exception = "Except Gemini"
        #     else:
        #         exception += ", Gemini"

        # Third case: OpenAI API
        res = self.openai_api.connect()
        if res:
            self.answers["openai"] = []
            for data in self.prompts[:prompt_size]:
                response = self.openai_api.generate(
                    prompt=data["prompt"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n_samples=n_samples,
                )
                self.answers["openai"].append(
                    {"response": response, "testbench": data["testbench"]}
                )
            with open("prompts-openai.json", "w") as fs:
                json.dump(self.answers["openai"], fs, indent=4)
            self.log.info(f"Written file for openai")
            filenames.append("prompts-openai.json")
        else:
            self.log.error(f"OpenAI connection unavailable; Moving to next case...")
            if len(exception) == 0:
                exception = "Except OpenAI"
            else:
                exception += ", OpenAI"

        # Last case: OSS LLM local API
        for model_info in self.models:
            oss_llm_api = OpenSourceLLMClient(
                model_id=model_info["id"], model_name=model_info["name"]
            )
            res = oss_llm_api.connect()
            if res:
                self.answers[model_info["id"]] = []
                for data in self.prompts[:prompt_size]:
                    response = self.oss_llm_api.generate(
                        prompt=data["prompt"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n_samples=n_samples,
                    )
                    self.answers[model_info["id"]].append(
                        {"response": response, "testbench": data["testbench"]}
                    )
                with open(f"prompts-{model_info['id']}.json", "w") as fs:
                    json.dump(self.answers[model_info["id"]], fs, indent=4)
                self.log.info(f"Written file for {model_info['id']}")
                filenames.append(f"prompts-{model_info['id']}.json")
            else:
                self.log.error(
                    f"{model_info['id']} connection unavailable; Moving to next case..."
                )
                if len(exception) == 0:
                    exception = f"Except {model_info['id']}"
                else:
                    exception += f", {model_info['id']}"

        if len(exception) == 0:
            with open(f"prompts-all.json", "w") as fs:
                json.dump(self.answers, fs, indent=4)
            # filenames = []
            filenames.append(f"prompts-all.json")

        self.log.info(f"Collected all prompts!")
        answers = self.answers
        return (answers, filenames)


if __name__ == "__main__":
    args = sys.argv()
    log = Logger("main").get_logger()
    gcp_storage = GoogleStorageClient()
    gcp_storage.connect()

    if len(args) == 0:
        log.critical("No arguments passed")

    elif len(args) == 1 and args[0] == "prompt":
        env = ENVLoader()
        runner = RunLLMPrompts()
        runner.collect_prompts()  # collect all prompts
        _, filenames = runner.run_all_models()  # run for all models
        for filename in filenames:
            try:
                gcp_storage.upload_file(
                    local_file_path=filename,
                    bucket_name="verilog-llm-eval",  # could change in your machine
                    blob_name=f"results/{filename}",
                )
                log.info(f"Copied file {filename} to GCP storage: results/")
            except Exception as err:
                log.error(f"Error in uploading file: {err}")
    else:
        log.warning("Unknown argument used")

    # Finally copy all log files
    log_filenames = os.listdir("logs/")
    for log_filename in log_filenames:
        try:
            gcp_storage.upload_file(
                local_file_path=filename,
                bucket_name="verilog-llm-eval",  # could change in your machine
                blob_name=f"logs/{filename}",
            )
            print(f"Copied file {filename} to GCP storage: logs/")
        except Exception as err:
            print(f"Error in uploading file: {err}")
