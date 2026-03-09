"""
IFEval Metric - Class-based implementation for instruction following evaluation
Based on google-research/instruction_following_eval
"""
# %%
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import collections
from final_metric_refactor.hard_gate.ifeval_core import instructions_registry


@dataclass
class EvaluationResult:
    """Result for a single response evaluation"""
    response: str
    prompt_level_strict_acc: int
    prompt_level_loose_acc: int
    inst_level_strict_acc: List[bool]
    inst_level_loose_acc: List[bool]
    inst_level_strict_acc_dict: Dict[str, bool]
    inst_level_loose_acc_dict: Dict[str, bool]
    instruction_results: List[Dict[str, Any]]
    total_instructions_tested: int


@dataclass
class AggregatedResult:
    """Aggregated results for multiple responses"""
    num_responses: int
    eval_mode: str
    prompt_level_strict_acc_mean: float
    prompt_level_loose_acc_mean: float
    inst_level_strict_acc_mean: Dict[str, float]
    inst_level_loose_acc_mean: Dict[str, float]
    individual_results: List[EvaluationResult]


class IFEvalMetric:
    """
    IFEval Metric class for evaluating instruction following.

    This class evaluates whether model responses follow specific instructions
    using strict and/or loose evaluation modes.

    Usage:
        metric = IFEvalMetric()
        result = metric.evaluate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Write a response in JSON format.",
            output=[{"response": "{'key': 'value'}"}]
        )
    """

    # Default instructions to evaluate
    DEFAULT_INSTRUCTIONS = [
        "detectable_format:json_format",
        "keywords:forbidden_words",
    ]

    def __init__(
        self,
        instruction_id_list: Optional[List[str]] = None,
        kwargs_list: Optional[List[Dict]] = None,
        eval_mode: str = "strict",
    ):
        """
        Initialize the IFEval metric.

        Args:
            instruction_id_list: List of instruction IDs to evaluate.
                                 If None, uses DEFAULT_INSTRUCTIONS (json_format, forbidden_words).
            kwargs_list: List of kwargs for each instruction.
                         If None, empty kwargs will be used.
            eval_mode: Evaluation mode - "strict", "loose", or "both". Default is "strict".
        """
        self.instruction_id_list = instruction_id_list
        self.kwargs_list = kwargs_list
        self._available_instructions = list(instructions_registry.INSTRUCTION_DICT.keys())

        if eval_mode not in ("strict", "loose", "both"):
            raise ValueError(f"eval_mode must be 'strict', 'loose', or 'both', got '{eval_mode}'")
        self.eval_mode = eval_mode

    @property
    def available_instructions(self) -> List[str]:
        """Return list of all available instruction IDs"""
        return self._available_instructions

    def _prepare_response_variants(self, response: str) -> List[str]:
        """Prepare response variants for loose evaluation"""
        r = response.split("\n")
        response_remove_first = "\n".join(r[1:]).strip()
        response_remove_last = "\n".join(r[:-1]).strip()
        response_remove_both = "\n".join(r[1:-1]).strip()
        revised_response = response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")

        return [
            response,
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            revised_response_remove_first,
            revised_response_remove_last,
            revised_response_remove_both,
        ]

    def _evaluate_single_response(
        self,
        response: str,
        prompt: str,
        instruction_id_list: List[str],
        kwargs_list: List[Dict]
    ) -> EvaluationResult:
        """
        Evaluate a single response against instructions.

        Args:
            response: The model response to evaluate
            prompt: The full prompt (system + user)
            instruction_id_list: List of instruction IDs to check
            kwargs_list: List of kwargs for each instruction

        Returns:
            EvaluationResult with detailed evaluation metrics
        """
        all_responses = self._prepare_response_variants(response)

        is_following_list_strict = []
        is_following_list_loose = []
        instruction_results = []

        for index, instruction_id in enumerate(instruction_id_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            task_kwargs = {k: v for k, v in kwargs_list[index].items() if v is not None}

            try:
                instruction.build_description(**task_kwargs)
                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    instruction.build_description(prompt=prompt)

                # Strict evaluation
                strict_result = bool(response.strip() and instruction.check_following(response))
                is_following_list_strict.append(strict_result)

                # Loose evaluation
                loose_result = False
                for r_variant in all_responses:
                    if r_variant.strip() and instruction.check_following(r_variant):
                        loose_result = True
                        break
                is_following_list_loose.append(loose_result)

                instruction_results.append({
                    "instruction_id": instruction_id,
                    "kwargs": task_kwargs,
                    "strict_passed": strict_result,
                    "loose_passed": loose_result,
                    "error": None
                })
            except Exception as e:
                is_following_list_strict.append(False)
                is_following_list_loose.append(False)
                instruction_results.append({
                    "instruction_id": instruction_id,
                    "kwargs": task_kwargs,
                    "strict_passed": False,
                    "loose_passed": False,
                    "error": str(e)
                })

        # Create instruction name: True/False dictionaries
        inst_level_strict_acc_dict = {
            instruction_id_list[i]: is_following_list_strict[i]
            for i in range(len(instruction_id_list))
        }
        inst_level_loose_acc_dict = {
            instruction_id_list[i]: is_following_list_loose[i]
            for i in range(len(instruction_id_list))
        }

        return EvaluationResult(
            response=response,
            prompt_level_strict_acc=int(all(is_following_list_strict)) if is_following_list_strict else 0,
            prompt_level_loose_acc=int(all(is_following_list_loose)) if is_following_list_loose else 0,
            inst_level_strict_acc=is_following_list_strict,
            inst_level_loose_acc=is_following_list_loose,
            inst_level_strict_acc_dict=inst_level_strict_acc_dict,
            inst_level_loose_acc_dict=inst_level_loose_acc_dict,
            instruction_results=instruction_results,
            total_instructions_tested=len(instruction_id_list)
        )

    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        output: List[Dict[str, Any]],
        response_key: str = "response"
    ) -> AggregatedResult:
        """
        Main evaluation method.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            output: List of JSON output dictionaries, each containing a response
            response_key: The key in output dicts that contains the response text

        Returns:
            AggregatedResult with evaluation metrics for all responses
        """
        # Build full prompt
        if system_prompt and user_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = system_prompt or user_prompt

        # Determine instructions to evaluate (use defaults if not specified)
        instruction_id_list = self.instruction_id_list
        if instruction_id_list is None:
            instruction_id_list = self.DEFAULT_INSTRUCTIONS

        kwargs_list = self.kwargs_list
        if kwargs_list is None:
            kwargs_list = [{} for _ in instruction_id_list]

        # Evaluate each response
        individual_results = []
        for output_item in output:
            if isinstance(output_item, dict):
                response = str(output_item.get(response_key, ""))
            else:
                response = str(output_item)

            result = self._evaluate_single_response(
                response=response,
                prompt=full_prompt,
                instruction_id_list=instruction_id_list,
                kwargs_list=kwargs_list
            )
            individual_results.append(result)

        # Aggregate results
        if not individual_results:
            return AggregatedResult(
                num_responses=0,
                eval_mode=self.eval_mode,
                prompt_level_strict_acc_mean=0.0,
                prompt_level_loose_acc_mean=0.0,
                inst_level_strict_acc_mean={},
                inst_level_loose_acc_mean={},
                individual_results=[]
            )

        # Calculate mean accuracies
        prompt_strict_sum = sum(r.prompt_level_strict_acc for r in individual_results)
        prompt_loose_sum = sum(r.prompt_level_loose_acc for r in individual_results)
        num_responses = len(individual_results)

        # Calculate instruction-level mean accuracies
        inst_strict_acc_sums = collections.defaultdict(float)
        inst_loose_acc_sums = collections.defaultdict(float)
        inst_counts = collections.defaultdict(int)

        for result in individual_results:
            for inst_id, passed in result.inst_level_strict_acc_dict.items():
                inst_strict_acc_sums[inst_id] += int(passed)
                inst_counts[inst_id] += 1
            for inst_id, passed in result.inst_level_loose_acc_dict.items():
                inst_loose_acc_sums[inst_id] += int(passed)

        inst_level_strict_acc_mean = {
            inst_id: inst_strict_acc_sums[inst_id] / inst_counts[inst_id]
            for inst_id in inst_counts
        }
        inst_level_loose_acc_mean = {
            inst_id: inst_loose_acc_sums[inst_id] / inst_counts[inst_id]
            for inst_id in inst_counts
        }

        return AggregatedResult(
            num_responses=num_responses,
            eval_mode=self.eval_mode,
            prompt_level_strict_acc_mean=prompt_strict_sum / num_responses,
            prompt_level_loose_acc_mean=prompt_loose_sum / num_responses,
            inst_level_strict_acc_mean=inst_level_strict_acc_mean,
            inst_level_loose_acc_mean=inst_level_loose_acc_mean,
            individual_results=individual_results
        )

    def evaluate_single(
        self,
        system_prompt: str,
        user_prompt: str,
        response: str
    ) -> EvaluationResult:
        """
        Convenience method to evaluate a single response.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            response: The model response text

        Returns:
            EvaluationResult for the single response
        """
        result = self.evaluate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output=[{"response": response}]
        )
        return result.individual_results[0] if result.individual_results else None


if __name__ == "__main__":
    # Example usage with defaults (json_format, forbidden_words, strict mode)
    print("Default instructions:", IFEvalMetric.DEFAULT_INSTRUCTIONS)

    metric = IFEvalMetric()
    print(f"\nDefault eval_mode: {metric.eval_mode}")

    print("\nAvailable instructions:")
    for inst_id in metric.available_instructions:
        print(f"  - {inst_id}")

    # Test with default settings (strict mode)
    print("\n" + "=" * 50)
    print("Test with defaults (strict mode)")
    print("=" * 50)

    result = metric.evaluate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Please respond in JSON format.",
        output=[
            {"response": '{"message": "Hello, world!"}'},
            {"response": "This is not JSON format."}
        ]
    )

    print(f"Eval mode: {result.eval_mode}")
    print(f"Number of responses: {result.num_responses}")
    print(f"Prompt-level strict accuracy: {result.prompt_level_strict_acc_mean:.2f}")

    # Test with both modes
    print("\n" + "=" * 50)
    print("Test with eval_mode='both'")
    print("=" * 50)

    metric_both = IFEvalMetric(eval_mode="both")
    result_both = metric_both.evaluate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Please respond in JSON format.",
        output=[
            {"response": '{"message": "Hello, world!"}'},
            {"response": "This is not JSON format."}
        ]
    )

    print(f"Eval mode: {result_both.eval_mode}")
    print(f"Prompt-level strict accuracy: {result_both.prompt_level_strict_acc_mean:.2f}")
    print(f"Prompt-level loose accuracy: {result_both.prompt_level_loose_acc_mean:.2f}")
