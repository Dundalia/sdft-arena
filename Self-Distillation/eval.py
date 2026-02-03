"""
Evaluation Script for Self-Distillation Models

This script evaluates trained models on various test sets using vLLM for efficient inference.
Supports MMLU, ToolUse, and future datasets.

Usage:
    Single dataset:
        python eval.py evaluation.evaluate_on=mmlu
    
    All datasets:
        python eval.py evaluation.evaluate_on=all
    
    Specific checkpoint:
        python eval.py model.checkpoint_path=/path/to/checkpoint
    
    Limit samples:
        python eval.py evaluation.max_samples=100
"""

import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import re

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvalSample:
    """A single evaluation sample."""
    id: int
    prompt: str
    reference: Any  # Ground truth answer
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result for a single evaluation sample."""
    id: int
    prompt: str
    generated: str
    reference: Any
    is_correct: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetStats:
    """Aggregated statistics for a dataset."""
    dataset_name: str
    total_samples: int
    correct: int
    accuracy: float
    per_category_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    additional_metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Dataset Evaluators (Modular Design for Future Extensions)
# =============================================================================

class BaseDatasetEvaluator(ABC):
    """Abstract base class for dataset-specific evaluators."""
    
    def __init__(self, data_path: Path, max_samples: Optional[int] = None, seed: int = 42):
        self.data_path = data_path
        self.max_samples = max_samples
        self.seed = seed
        self.samples: List[EvalSample] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        pass
    
    @abstractmethod
    def load_data(self) -> List[EvalSample]:
        """Load and prepare evaluation samples."""
        pass
    
    @abstractmethod
    def format_prompt(self, sample: EvalSample) -> str:
        """Format a sample into a prompt for the model."""
        pass
    
    @abstractmethod
    def evaluate_response(self, sample: EvalSample, generated: str) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a generated response against the reference."""
        pass
    
    def compute_stats(self, results: List[EvalResult]) -> DatasetStats:
        """Compute aggregated statistics from results."""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0.0
        
        return DatasetStats(
            dataset_name=self.name,
            total_samples=total,
            correct=correct,
            accuracy=accuracy
        )


class MMLUEvaluator(BaseDatasetEvaluator):
    """Evaluator for the MMLU (Massive Multitask Language Understanding) dataset."""
    
    CHOICE_LETTERS = ['A', 'B', 'C', 'D']
    
    @property
    def name(self) -> str:
        return "mmlu"
    
    def load_data(self) -> List[EvalSample]:
        """Load MMLU data from JSONL format."""
        samples = []
        eval_file = self.data_path / "eval_data.json"
        
        if not eval_file.exists():
            raise FileNotFoundError(f"MMLU eval data not found at {eval_file}")
        
        with open(eval_file, 'r') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    samples.append(EvalSample(
                        id=idx,
                        prompt=data['question'],
                        reference=data['answer'],  # Integer index
                        metadata={
                            'subject': data.get('subject', 'unknown'),
                            'choices': data['choices']
                        }
                    ))
        
        # Shuffle and limit samples if needed
        import random
        random.seed(self.seed)
        random.shuffle(samples)
        
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        
        self.samples = samples
        return samples
    
    def format_prompt(self, sample: EvalSample) -> str:
        """Format MMLU sample as multiple choice question."""
        choices = sample.metadata['choices']
        choices_str = '\n'.join(
            f"{letter}. {choice}" 
            for letter, choice in zip(self.CHOICE_LETTERS, choices)
        )
        
        prompt = f"""Answer the following multiple choice question. Reply with only the letter (A, B, C, or D) of the correct answer.

Question: {sample.prompt}

{choices_str}

Answer:"""
        return prompt
    
    def evaluate_response(self, sample: EvalSample, generated: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if the generated response matches the correct answer."""
        # Extract the answer letter from the generated text
        generated_clean = generated.strip().upper()
        
        # Try to find a letter answer (A, B, C, or D)
        predicted_idx = None
        
        # Check for patterns like "A", "A.", "(A)", "The answer is A", etc.
        for i, letter in enumerate(self.CHOICE_LETTERS):
            patterns = [
                rf'^{letter}[.\s\)]',  # Starts with letter
                rf'^{letter}$',  # Just the letter
                rf'answer is[:\s]*{letter}',  # "answer is A"
                rf'\({letter}\)',  # (A)
            ]
            for pattern in patterns:
                if re.search(pattern, generated_clean, re.IGNORECASE):
                    predicted_idx = i
                    break
            if predicted_idx is not None:
                break
        
        # Fallback: just check first non-whitespace character
        if predicted_idx is None and generated_clean:
            first_char = generated_clean[0]
            if first_char in self.CHOICE_LETTERS:
                predicted_idx = self.CHOICE_LETTERS.index(first_char)
        
        is_correct = predicted_idx == sample.reference
        
        return is_correct, {
            'predicted_idx': predicted_idx,
            'predicted_letter': self.CHOICE_LETTERS[predicted_idx] if predicted_idx is not None else None,
            'correct_letter': self.CHOICE_LETTERS[sample.reference],
            'subject': sample.metadata['subject']
        }
    
    def compute_stats(self, results: List[EvalResult]) -> DatasetStats:
        """Compute stats including per-subject breakdown."""
        base_stats = super().compute_stats(results)
        
        # Per-subject statistics
        subject_results: Dict[str, List[bool]] = {}
        for r in results:
            subject = r.metadata.get('subject', 'unknown')
            if subject not in subject_results:
                subject_results[subject] = []
            subject_results[subject].append(r.is_correct)
        
        per_category_stats = {}
        for subject, correctness in subject_results.items():
            total = len(correctness)
            correct = sum(correctness)
            per_category_stats[subject] = {
                'total': total,
                'correct': correct,
                'accuracy': correct / total if total > 0 else 0.0
            }
        
        base_stats.per_category_stats = per_category_stats
        return base_stats


class ToolUseEvaluator(BaseDatasetEvaluator):
    """Evaluator for the Tool Use dataset."""
    
    @property
    def name(self) -> str:
        return "tooluse_data"
    
    def load_data(self) -> List[EvalSample]:
        """Load ToolUse data from JSON format."""
        samples = []
        eval_file = self.data_path / "eval_data.json"
        
        if not eval_file.exists():
            raise FileNotFoundError(f"ToolUse eval data not found at {eval_file}")
        
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        for idx, item in enumerate(data):
            samples.append(EvalSample(
                id=idx,
                prompt=item['prompt'],
                reference=item['golden_answer'],  # List of action dicts
                metadata={
                    'name': item.get('name', ''),
                    'description': item.get('description', ''),
                    'instruction': item.get('instruction', '')
                }
            ))
        
        # Shuffle and limit samples if needed
        import random
        random.seed(self.seed)
        random.shuffle(samples)
        
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        
        self.samples = samples
        return samples
    
    def format_prompt(self, sample: EvalSample) -> str:
        """Format ToolUse sample as a prompt."""
        # The prompt already contains the full instruction
        return sample.prompt
    
    def evaluate_response(self, sample: EvalSample, generated: str) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate if the generated response contains the correct actions."""
        golden_actions = sample.reference
        
        # Extract actions from generated text
        extracted_actions = self._extract_actions(generated)
        
        # Check if all golden actions are present (order-independent for now)
        correct_actions = 0
        total_golden = len(golden_actions)
        
        matched_actions = []
        for golden in golden_actions:
            golden_action = golden.get('Action', '')
            for extracted in extracted_actions:
                if extracted.get('Action', '').lower() == golden_action.lower():
                    correct_actions += 1
                    matched_actions.append(golden_action)
                    break
        
        # Consider correct if all golden actions are found
        is_correct = correct_actions == total_golden and total_golden > 0
        
        # Partial credit metric
        action_recall = correct_actions / total_golden if total_golden > 0 else 0.0
        
        return is_correct, {
            'extracted_actions': extracted_actions,
            'golden_actions': golden_actions,
            'matched_actions': matched_actions,
            'action_recall': action_recall,
            'tool_name': sample.metadata.get('name', '')
        }
    
    def _extract_actions(self, text: str) -> List[Dict[str, str]]:
        """Extract action-input pairs from generated text."""
        actions = []
        
        # Pattern to match "Action: <action_name>" and "Action Input: <json>"
        action_pattern = r'Action:\s*(\w+)'
        input_pattern = r'Action Input:\s*({[^}]+}|\{[^}]*\})'
        
        action_matches = re.findall(action_pattern, text, re.IGNORECASE)
        input_matches = re.findall(input_pattern, text, re.IGNORECASE)
        
        for i, action in enumerate(action_matches):
            action_dict = {'Action': action}
            if i < len(input_matches):
                action_dict['Action_Input'] = input_matches[i]
            actions.append(action_dict)
        
        return actions
    
    def compute_stats(self, results: List[EvalResult]) -> DatasetStats:
        """Compute stats including action recall metrics."""
        base_stats = super().compute_stats(results)
        
        # Compute average action recall
        recalls = [r.metadata.get('action_recall', 0.0) for r in results]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        
        # Per-tool statistics
        tool_results: Dict[str, List[bool]] = {}
        for r in results:
            tool = r.metadata.get('tool_name', 'unknown')
            if tool not in tool_results:
                tool_results[tool] = []
            tool_results[tool].append(r.is_correct)
        
        per_category_stats = {}
        for tool, correctness in list(tool_results.items())[:20]:  # Limit to top 20 tools
            total = len(correctness)
            correct = sum(correctness)
            per_category_stats[tool] = {
                'total': total,
                'correct': correct,
                'accuracy': correct / total if total > 0 else 0.0
            }
        
        base_stats.per_category_stats = per_category_stats
        base_stats.additional_metrics = {
            'avg_action_recall': avg_recall,
            'exact_match_rate': base_stats.accuracy
        }
        
        return base_stats


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_EVALUATORS = {
    'mmlu': MMLUEvaluator,
    'tooluse_data': ToolUseEvaluator,
}


def get_available_datasets(data_dir: Path) -> List[str]:
    """Get list of available datasets in the data directory."""
    available = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and (subdir / "eval_data.json").exists():
            available.append(subdir.name)
    return available


def get_evaluator(dataset_name: str, data_dir: Path, max_samples: Optional[int], seed: int) -> BaseDatasetEvaluator:
    """Get the appropriate evaluator for a dataset."""
    data_path = data_dir / dataset_name
    
    if dataset_name in DATASET_EVALUATORS:
        return DATASET_EVALUATORS[dataset_name](data_path, max_samples, seed)
    else:
        # Default to a generic evaluator if available, or raise error
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available evaluators: {list(DATASET_EVALUATORS.keys())}. "
            f"To add support for new datasets, implement a BaseDatasetEvaluator subclass."
        )


# =============================================================================
# Main Evaluation Logic
# =============================================================================

class Evaluator:
    """Main evaluator class that handles model loading and inference."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None
        self.results_dir = None
    
    def setup(self):
        """Setup the evaluator: load model and prepare output directory."""
        print("=" * 60)
        print("Setting up Evaluator")
        print("=" * 60)
        
        # Setup results directory
        script_dir = Path(__file__).parent
        results_base = script_dir / "results"
        self.results_dir = results_base / self.cfg.output.results_subdir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.results_dir}")
        
        # Load tokenizer
        print(f"Loading tokenizer from: {self.cfg.model.checkpoint_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.checkpoint_path)
        
        # Load model with vLLM
        print(f"Loading model with vLLM...")
        print(f"  - Tensor parallel size: {self.cfg.vllm.tensor_parallel_size}")
        print(f"  - GPU memory utilization: {self.cfg.vllm.gpu_memory_utilization}")
        
        self.model = LLM(
            model=self.cfg.model.checkpoint_path,
            tensor_parallel_size=self.cfg.vllm.tensor_parallel_size,
            gpu_memory_utilization=self.cfg.vllm.gpu_memory_utilization,
            dtype=self.cfg.model.torch_dtype,
            trust_remote_code=True,
        )
        
        print("Model loaded successfully!")
    
    def run_inference(self, prompts: List[str]) -> List[str]:
        """Run inference on a batch of prompts using vLLM."""
        sampling_params = SamplingParams(
            temperature=self.cfg.vllm.temperature,
            top_p=self.cfg.vllm.top_p,
            max_tokens=self.cfg.vllm.max_new_tokens,
        )
        
        # Format prompts with chat template if available
        formatted_prompts = []
        for prompt in prompts:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
            else:
                formatted_prompts.append(prompt)
        
        # Run inference
        outputs = self.model.generate(formatted_prompts, sampling_params)
        
        # Extract generated text
        generated_texts = []
        for output in outputs:
            generated_texts.append(output.outputs[0].text)
        
        return generated_texts
    
    def evaluate_dataset(self, evaluator: BaseDatasetEvaluator) -> Tuple[DatasetStats, List[EvalResult]]:
        """Evaluate a single dataset."""
        print(f"\n{'=' * 60}")
        print(f"Evaluating dataset: {evaluator.name}")
        print(f"{'=' * 60}")
        
        # Load data
        samples = evaluator.load_data()
        print(f"Loaded {len(samples)} samples")
        
        # Prepare prompts
        prompts = [evaluator.format_prompt(sample) for sample in samples]
        
        # Run inference in batches for progress tracking
        batch_size = self.cfg.evaluation.batch_size
        all_generated = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            if self.cfg.logging.verbose:
                print(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")
            
            generated = self.run_inference(batch_prompts)
            all_generated.extend(generated)
        
        # Evaluate responses
        results = []
        for sample, generated in zip(samples, all_generated):
            is_correct, eval_metadata = evaluator.evaluate_response(sample, generated)
            results.append(EvalResult(
                id=sample.id,
                prompt=sample.prompt,
                generated=generated,
                reference=sample.reference,
                is_correct=is_correct,
                metadata=eval_metadata
            ))
        
        # Compute statistics
        stats = evaluator.compute_stats(results)
        
        print(f"\n{evaluator.name} Results:")
        print(f"  Total samples: {stats.total_samples}")
        print(f"  Correct: {stats.correct}")
        print(f"  Accuracy: {stats.accuracy:.4f} ({stats.accuracy * 100:.2f}%)")
        
        if stats.additional_metrics:
            print(f"  Additional metrics:")
            for key, value in stats.additional_metrics.items():
                print(f"    {key}: {value:.4f}")
        
        return stats, results
    
    def save_results(self, dataset_name: str, stats: DatasetStats, results: List[EvalResult]):
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.results_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save statistics
        if self.cfg.output.save_stats:
            stats_file = dataset_dir / f"stats_{timestamp}.json"
            stats_dict = {
                'dataset_name': stats.dataset_name,
                'total_samples': stats.total_samples,
                'correct': stats.correct,
                'accuracy': stats.accuracy,
                'per_category_stats': stats.per_category_stats,
                'additional_metrics': stats.additional_metrics,
                'model_checkpoint': self.cfg.model.checkpoint_path,
                'timestamp': timestamp,
                'config': OmegaConf.to_container(self.cfg, resolve=True)
            }
            with open(stats_file, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            print(f"Stats saved to: {stats_file}")
        
        # Save predictions
        if self.cfg.output.save_predictions:
            predictions_file = dataset_dir / f"predictions_{timestamp}.json"
            predictions = []
            for r in results:
                predictions.append({
                    'id': r.id,
                    'prompt': r.prompt[:500] + '...' if len(r.prompt) > 500 else r.prompt,
                    'generated': r.generated,
                    'reference': str(r.reference)[:500] if len(str(r.reference)) > 500 else r.reference,
                    'is_correct': r.is_correct,
                    'metadata': r.metadata
                })
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Predictions saved to: {predictions_file}")
    
    def run(self):
        """Run the full evaluation pipeline."""
        self.setup()
        
        # Get data directory
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
        
        # Determine which datasets to evaluate
        if self.cfg.evaluation.evaluate_on == "all":
            datasets = get_available_datasets(data_dir)
            print(f"\nFound {len(datasets)} datasets to evaluate: {datasets}")
        else:
            datasets = [self.cfg.evaluation.evaluate_on]
        
        # Evaluate each dataset
        all_stats = {}
        for dataset_name in datasets:
            try:
                evaluator = get_evaluator(
                    dataset_name,
                    data_dir,
                    self.cfg.evaluation.max_samples,
                    self.cfg.evaluation.seed
                )
                stats, results = self.evaluate_dataset(evaluator)
                all_stats[dataset_name] = stats
                
                # Save results
                self.save_results(dataset_name, stats, results)
                
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        for dataset_name, stats in all_stats.items():
            print(f"{dataset_name}: {stats.accuracy:.4f} ({stats.correct}/{stats.total_samples})")
        
        # Save combined summary
        summary_file = self.results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary = {
            'model_checkpoint': self.cfg.model.checkpoint_path,
            'datasets': {
                name: {
                    'accuracy': stats.accuracy,
                    'correct': stats.correct,
                    'total': stats.total_samples
                }
                for name, stats in all_stats.items()
            }
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")


# =============================================================================
# Main Entry Point
# =============================================================================

@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg: DictConfig):
    """Main evaluation function with Hydra configuration."""
    
    # Print configuration
    print("=== Evaluation Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    print("================================")
    
    # Change to script directory for relative paths
    os.chdir(Path(__file__).parent)
    
    # Run evaluation
    evaluator = Evaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
