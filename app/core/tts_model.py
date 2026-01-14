"""
TTS model initialization and management
"""

import os
import asyncio
from enum import Enum
from typing import Optional, Dict, Any
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from app.core.mtl import SUPPORTED_LANGUAGES
from app.config import Config, detect_device
import huggingface_hub

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
if HUGGINGFACE_TOKEN:
    huggingface_hub.login(token=HUGGINGFACE_TOKEN)

# Global model instance
_model = None
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_is_multilingual = None
_supported_languages = {}


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


# from torch.ao.quantization import quantize_dynamic
# def quantize_model(model: ChatterboxTTS):
#     # Placeholder for quantization logic if needed in future
#     model.t3 = quantize_dynamic(
#         model.t3, 
#         {torch.nn.Linear}, 
#         dtype=torch.qint8
#     )
    
#     # Quantize S3 model to INT8
#     model.s3 = quantize_dynamic(
#         model.s3,
#         {torch.nn.Linear},
#         dtype=torch.qint8
#     )
    
#     # Force contiguous memory layout
#     for param in model.parameters():
#         if param.data.is_cuda:
#             param.data = param.data.contiguous()
    
#     return model


async def initialize_model():
    """Initialize the Chatterbox TTS model"""
    global _model, _device, _initialization_state, _initialization_error, _initialization_progress, _is_multilingual, _supported_languages
    
    try:
        def t3_to(model: ChatterboxTTS, dtype):
            model.t3.to(dtype=dtype)
            model.conds.t3.to(dtype=dtype)
            torch.cuda.empty_cache()
            return model
        
        def optimize_model(model: ChatterboxTTS):
            import torch
            print("[optimize_model] Starting optimization...")
            # 1. Move both T3 and S3 to bfloat16
            print("[optimize_model] Moving model.t3 to bfloat16...")
            model.t3.to(dtype=torch.bfloat16)
            print("[optimize_model] Moving model.conds.t3 to bfloat16...")
            model.conds.t3.to(dtype=torch.bfloat16)
            # print("[optimize_model] Moving model.s3 to bfloat16...")
            # model.s3.to(dtype=torch.bfloat16)  # Add S3 optimization

            # 2. Pre-compile CUDA graphs for common operations
            print("[optimize_model] Checking for CUDA availability...")
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                print("[optimize_model] CUDA is available. Emptying CUDA cache...")
                torch.cuda.empty_cache()
                print("[optimize_model] Enabling CUDA graphs via torch._inductor.config.triton.cudagraphs...")
                torch._inductor.config.triton.cudagraphs = True
            else:
                print("[optimize_model] CUDA not available, skipping CUDA-specific optimizations.")

            # 3. Set model to eval mode explicitly
            print("[optimize_model] Setting model.t3 to eval mode...")
            model.t3.eval()
            # print("[optimize_model] Setting model.s3 to eval mode...")
            # model.s3.eval()

            # 4. Optimize memory usage
            print("[optimize_model] Checking for CUDA memory optimization...")
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                print("[optimize_model] Increasing GPU memory fraction to 0.9...")
                torch.cuda.set_per_process_memory_fraction(0.9)
            else:
                print("[optimize_model] CUDA not available, skipping GPU memory fraction adjustment.")

            print("[optimize_model] Optimization complete.")
            return model
        
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."
        
        Config.validate()
        _device = detect_device()
        
        print(f"Initializing Chatterbox TTS model...")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")
        
        _initialization_progress = "Creating model cache directory..."
        # Ensure model cache directory exists
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        _initialization_progress = "Checking voice sample..."
        # Check voice sample exists
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}")
        
        _initialization_progress = "Configuring device compatibility..."
        # Patch torch.load for CPU compatibility if needed
        if _device == 'cpu':
            import torch
            original_load = torch.load
            original_load_file = None
            
            # Try to patch safetensors if available
            try:
                import safetensors.torch
                original_load_file = safetensors.torch.load_file
            except ImportError:
                pass
            
            def force_cpu_torch_load(f, map_location=None, **kwargs):
                # Always force CPU mapping if we're on a CPU device
                return original_load(f, map_location='cpu', **kwargs)
            
            def force_cpu_load_file(filename, device=None):
                # Force CPU for safetensors loading too
                return original_load_file(filename, device='cpu')
            
            torch.load = force_cpu_torch_load
            if original_load_file:
                safetensors.torch.load_file = force_cpu_load_file
        
        # Determine if we should use multilingual model
        use_multilingual = Config.USE_MULTILINGUAL_MODEL
        
        _initialization_progress = "Loading TTS model (this may take a while)..."
        # Initialize model with run_in_executor for non-blocking
        loop = asyncio.get_event_loop()
        
        if use_multilingual:
            print(f"Loading Chatterbox Multilingual TTS model...")
            # _model = await loop.run_in_executor(
            #     None, 
            #     lambda: ChatterboxMultilingualTTS.from_pretrained(device=_device)
            # )
            huggingface_hub.login(token=HUGGINGFACE_TOKEN)  # Replace with your actual token or use environment variable
            _model = await loop.run_in_executor(
                None, 
                lambda: ChatterboxTurboTTS.from_pretrained(device=_device)
            )
            _is_multilingual = True
            _supported_languages = SUPPORTED_LANGUAGES.copy()
            print(f"✓ Multilingual model initialized with {len(_supported_languages)} languages")
        else:
            print(f"Loading standard Chatterbox TTS model...")
            _model = await loop.run_in_executor(
                None, 
                lambda: ChatterboxTTS.from_pretrained(device=_device)
            )
            _is_multilingual = False
            _supported_languages = {"en": "English"}  # Standard model only supports English
            print(f"✓ Standard model initialized (English only)")

        # t3_to(_model, torch.bfloat16)
        _model = optimize_model(_model)
        print("Model has been optimized for performance!")
        # _model = quantize_model(_model)
        
        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"✓ Model initialized successfully on {_device}")
        return _model
        
    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"✗ Failed to initialize model: {e}")
        raise e


def get_model():
    """Get the current model instance"""
    return _model


def get_device():
    """Get the current device"""
    return _device


def get_initialization_state():
    """Get the current initialization state"""
    return _initialization_state


def get_initialization_progress():
    """Get the current initialization progress message"""
    return _initialization_progress


def get_initialization_error():
    """Get the initialization error if any"""
    return _initialization_error


def is_ready():
    """Check if the model is ready for use"""
    return _initialization_state == InitializationState.READY.value and _model is not None


def is_initializing():
    """Check if the model is currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value 


def is_multilingual():
    """Check if the loaded model supports multilingual generation"""
    return _is_multilingual


def get_supported_languages():
    """Get the dictionary of supported languages"""
    return _supported_languages.copy()


def supports_language(language_id: str):
    """Check if the model supports a specific language"""
    return language_id in _supported_languages


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information"""
    return {
        "model_type": "multilingual" if _is_multilingual else "standard",
        "is_multilingual": _is_multilingual,
        "supported_languages": _supported_languages,
        "language_count": len(_supported_languages),
        "device": _device,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state
    }