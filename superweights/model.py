"""Core model analysis functionality."""

from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import numpy as np
from .utils import get_weight_type, get_layer_number


class TransformerAnalyzer:
    """Analyzer for transformer models to identify and analyze superweights."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the analyzer with a model.
        
        Args:
            model_name: Name or path of the Hugging Face model
            device: Device to load the model on
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Load the model and tokenizer."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
    def find_superweights(
        self,
        threshold: float = 1e-4,
        weight_type: str = "all",
        layers: Optional[List[int]] = None,
        criterion: str = "percentage"
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Find superweights in the model based on specified criteria.
        
        Args:
            threshold: Threshold for identifying superweights
            weight_type: Type of weights to analyze ("all" or specific type)
            layers: List of layer indices to analyze
            criterion: Criterion for selection ("percentage" or "absolute")
        
        Returns:
            Dictionary mapping weight names to list of (row, col) superweight positions
        
        Raises:
            ValueError: If arguments are invalid
        """
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
        if criterion not in ["percentage", "absolute"]:
            raise ValueError("Criterion must be 'percentage' or 'absolute'")
        
        superweights = {}
        
        for name, param in self.model.named_parameters():
            if not name.endswith("weight"):
                continue
            if "layernorm" in name or "norm" in name or "embed_tokens" in name or "lm_head" in name:
                continue
                
            if weight_type != "all":
                current_weight_type = get_weight_type(self.model_name, name)
                if current_weight_type != weight_type:
                    continue
                    
            if layers is not None:
                layer_num = get_layer_number(name)
                if layer_num not in layers:
                    continue
                    
            weight = param.data
            if criterion == "percentage":
                num_elements = max(1, int(weight.numel() * threshold))
                flat_weight = weight.view(-1)
                _, indices = torch.topk(torch.abs(flat_weight), num_elements)
                positions = [(idx.item() // weight.size(1), idx.item() % weight.size(1)) 
                            for idx in indices]
            else:  # absolute threshold
                positions = torch.where(torch.abs(weight) > threshold)
                positions = list(zip(positions[0].cpu().numpy(), 
                                   positions[1].cpu().numpy()))
                
            if positions:
                superweights[name] = positions
                
        return superweights
    
    def remove_superweights(
        self,
        superweights: Dict[str, List[Tuple[int, int]]],
        method: str = "zero"
    ) -> None:
        """Remove or modify identified superweights.
        
        Args:
            superweights: Dictionary of superweight positions
            method: Method to handle superweights ("zero" or "mean")
        
        Raises:
            ValueError: If arguments are invalid
        """
        if not superweights:
            raise ValueError("Superweights dictionary cannot be empty")
        if method not in ["zero", "mean"]:
            raise ValueError("Method must be 'zero' or 'mean'")
            
        with torch.no_grad():
            for name, positions in superweights.items():
                param = dict(self.model.named_parameters())[name]
                if method == "zero":
                    for row, col in positions:
                        param.data[row, col] = 0.0
                elif method == "mean":
                    mean_value = param.data.mean()
                    for row, col in positions:
                        param.data[row, col] = mean_value

    def evaluate_impact(
        self,
        text: str,
        superweights: Dict[str, List[Tuple[int, int]]]
    ) -> Dict[str, float]:
        """Evaluate the impact of superweights on model output.
        
        Args:
            text: Input text for evaluation
            superweights: Dictionary of superweight positions
        
        Returns:
            Dictionary of metrics showing superweight impact
        
        Raises:
            ValueError: If arguments are invalid
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get original output
        with torch.no_grad():
            original_output = self.model(**inputs).logits
            
        # Remove superweights
        self.remove_superweights(superweights)
        
        # Get modified output
        with torch.no_grad():
            modified_output = self.model(**inputs).logits
            
        # Restore superweights by reinitializing model
        self._initialize_model()
        
        # Calculate impact metrics
        impact = {
            "output_diff_norm": torch.norm(original_output - modified_output).item(),
            "relative_change": (torch.norm(original_output - modified_output) / 
                              torch.norm(original_output)).item()
        }
        
        return impact

    def analyze_activations(
        self,
        text: str,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """Analyze activations for given input text.
        
        Args:
            text: Input text to analyze
            layer_indices: Specific layers to analyze
        
        Returns:
            Dictionary mapping layer indices to activation tensors
        
        Raises:
            ValueError: If text is empty or layer indices are invalid
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        activations = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    activations[layer_idx] = output[0].detach()
                else:
                    activations[layer_idx] = output.detach()
            return hook
        
        hooks = []
        if layer_indices is None:
            # Determine the number of layers based on model architecture
            if hasattr(self.model, "transformer"):
                num_layers = len(self.model.transformer.h)
                layer_indices = range(num_layers)
            elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                num_layers = len(self.model.model.layers)
                layer_indices = range(num_layers)
            else:
                raise ValueError("Unsupported model architecture")
        else:
            if not layer_indices:
                raise ValueError("Layer indices list cannot be empty")
            if min(layer_indices) < 0:
                raise ValueError("Layer indices must be non-negative")
            if hasattr(self.model, "transformer"):
                num_layers = len(self.model.transformer.h)
            elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                num_layers = len(self.model.model.layers)
            else:
                raise ValueError("Unsupported model architecture")
            if max(layer_indices) >= num_layers:
                raise ValueError(f"Layer index {max(layer_indices)} is out of range (max: {num_layers-1})")
                
        for idx in layer_indices:
            if hasattr(self.model, "transformer"):
                layer = self.model.transformer.h[idx]
            elif hasattr(self.model, "model"):
                layer = self.model.model.layers[idx]
            else:
                raise ValueError("Unsupported model architecture")
                
            hooks.append(layer.register_forward_hook(hook_fn(idx)))
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            self.model(**inputs)
            
        for hook in hooks:
            hook.remove()
            
        return activations

    def remove_superweights_and_get_modified_output(
        self,
        text: str,
        superweights: Dict[str, List[Tuple[int, int]]],
        method: str = "zero"
    ) -> torch.Tensor:
        """Remove or modify identified superweights and return modified output.
        
        Args:
            text: Input text to get modified output for
            superweights: Dictionary of superweight positions
            method: Method to handle superweights ("zero" or "mean")
        
        Returns:
            Modified model output tensor
        
        Raises:
            ValueError: If arguments are invalid
        """
        if not superweights:
            raise ValueError("Superweights dictionary cannot be empty")
        if method not in ["zero", "mean"]:
            raise ValueError("Method must be 'zero' or 'mean'")
            
        # Get original parameters
        original_params = {}
        for name, positions in superweights.items():
            param = dict(self.model.named_parameters())[name]
            original_params[name] = param.data.clone()
            
            if method == "zero":
                for row, col in positions:
                    param.data[row, col] = 0.0
            elif method == "mean":
                mean_value = param.data.mean()
                for row, col in positions:
                    param.data[row, col] = mean_value
                    
        # Get modified output
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            modified_output = self.model(**inputs).logits
            
        # Restore original parameters
        for name, param_data in original_params.items():
            dict(self.model.named_parameters())[name].data.copy_(param_data)
            
        return modified_output

    def evaluate_impact_with_modified_output(
        self,
        text: str,
        original_output: torch.Tensor,
        modified_output: torch.Tensor,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate the impact of superweights on model output.
        
        Args:
            text: Input text for evaluation
            original_output: Original model output
            modified_output: Modified model output (after removing superweights)
            metrics: List of metrics to compute (default: all available metrics)
        
        Returns:
            Dictionary of metrics showing superweight impact
        
        Raises:
            ValueError: If arguments are invalid
        """
        if metrics is None:
            metrics = ["output_diff_norm", "relative_change", "perplexity_change", "kl_divergence"]
        
        impact = {}
        
        if "output_diff_norm" in metrics:
            impact["output_diff_norm"] = torch.norm(original_output - modified_output).item()
            
        if "relative_change" in metrics:
            impact["relative_change"] = (torch.norm(original_output - modified_output) / 
                                       torch.norm(original_output)).item()
            
        if "perplexity_change" in metrics:
            original_probs = torch.nn.functional.softmax(original_output, dim=-1)
            modified_probs = torch.nn.functional.softmax(modified_output, dim=-1)
            original_entropy = -torch.sum(original_probs * torch.log(original_probs))
            modified_entropy = -torch.sum(modified_probs * torch.log(modified_probs))
            impact["perplexity_change"] = (torch.exp(modified_entropy) - 
                                         torch.exp(original_entropy)).item()
            
        if "kl_divergence" in metrics:
            original_probs = torch.nn.functional.softmax(original_output, dim=-1)
            modified_probs = torch.nn.functional.softmax(modified_output, dim=-1)
            impact["kl_divergence"] = torch.nn.functional.kl_div(
                modified_probs.log(), original_probs, reduction="batchmean"
            ).item()
            
        return impact
