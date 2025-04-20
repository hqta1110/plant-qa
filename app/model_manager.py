import time
import threading
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
import torch
import gc

# Type variable for generic model type
T = TypeVar('T')

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_manager")

class ModelManager(Generic[T]):
    """
    A manager for lazy loading and caching ML models.
    Models are loaded on demand and can be unloaded after a period of inactivity.
    """
    
    def __init__(
        self, 
        model_init_fn: Callable[[], T],
        model_name: str,
        max_idle_time: int = 3600,  # Default 1 hour idle time before unload
        device: str = None,
        priority: int = 1  # Higher priority models stay loaded longer (1-10)
    ):
        """
        Initialize the model manager.
        
        Args:
            model_init_fn: Function to initialize the model when needed
            model_name: Name identifier for the model
            max_idle_time: Maximum time (seconds) the model can be idle before unloading
            device: Device to load the model on (None for auto)
            priority: Higher priority models (1-10) stay in memory longer
        """
        self.model_init_fn = model_init_fn
        self.model_name = model_name
        self.max_idle_time = max_idle_time
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.priority = min(max(priority, 1), 10)  # Clamp between 1-10
        
        # Model state
        self._model: Optional[T] = None
        self._lock = threading.RLock()
        self._last_used = 0
        self._is_loaded = False
        self._is_loading = False
        self._load_error = None
        
        # Statistics
        self._load_count = 0
        self._use_count = 0
        
        # Register with global model registry for cleanup
        ModelRegistry.register(self)
        
        logger.info(f"Initialized model manager for {model_name} (priority: {priority})")
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded"""
        return self._is_loaded
    
    @property
    def idle_time(self) -> float:
        """Get the time (in seconds) since the model was last used"""
        if not self._is_loaded:
            return float('inf')
        return time.time() - self._last_used
    
    def get_model(self) -> T:
        """
        Get the model, loading it if necessary.
        
        Returns:
            The loaded model
            
        Raises:
            RuntimeError: If there was an error loading the model
        """
        with self._lock:
            # Update last used time
            self._last_used = time.time()
            self._use_count += 1
            
            # If model is already loaded, return it
            if self._is_loaded:
                return self._model
            
            # If there was a previous error loading the model, raise it
            if self._load_error:
                raise RuntimeError(f"Error loading model {self.model_name}: {self._load_error}")
            
            # If the model is currently being loaded by another thread, wait
            if self._is_loading:
                logger.info(f"Waiting for {self.model_name} to finish loading")
                while self._is_loading:
                    # Release lock while waiting
                    self._lock.release()
                    time.sleep(0.1)
                    self._lock.acquire()
                
                # If loaded successfully, return it
                if self._is_loaded:
                    return self._model
                # If loading failed, raise the error
                if self._load_error:
                    raise RuntimeError(f"Error loading model {self.model_name}: {self._load_error}")
            
            # Load the model
            try:
                logger.info(f"Loading model {self.model_name}")
                self._is_loading = True
                
                # Try to clean up memory before loading
                self._cleanup_memory()
                
                # Initialize the model
                self._model = self.model_init_fn()
                self._is_loaded = True
                self._load_count += 1
                
                logger.info(f"Model {self.model_name} loaded successfully")
                return self._model
            
            except Exception as e:
                self._load_error = str(e)
                logger.error(f"Error loading model {self.model_name}: {e}")
                raise RuntimeError(f"Error loading model {self.model_name}: {e}")
            
            finally:
                self._is_loading = False
    
    def unload(self) -> bool:
        """
        Unload the model from memory.
        
        Returns:
            True if the model was unloaded, False if it wasn't loaded
        """
        with self._lock:
            if not self._is_loaded:
                return False
            
            logger.info(f"Unloading model {self.model_name}")
            
            # Set model to None to release reference
            self._model = None
            self._is_loaded = False
            self._load_error = None
            
            # Force garbage collection
            self._cleanup_memory()
            
            return True
    
    def should_unload(self) -> bool:
        """Check if the model should be unloaded based on idle time"""
        # Calculate adjusted idle time based on priority
        adjusted_max_idle = self.max_idle_time * (self.priority / 5.0)
        return self.idle_time > adjusted_max_idle
    
    def _cleanup_memory(self):
        """Attempt to clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this model"""
        return {
            "name": self.model_name,
            "loaded": self._is_loaded,
            "idle_time": self.idle_time,
            "load_count": self._load_count,
            "use_count": self._use_count,
            "device": self.device,
            "priority": self.priority,
            "max_idle_time": self.max_idle_time
        }


class ModelRegistry:
    """
    Global registry to manage model instances and periodic cleanup.
    """
    _registry: Dict[str, ModelManager] = {}
    _lock = threading.RLock()
    _cleanup_thread: Optional[threading.Thread] = None
    _running = False
    
    @classmethod
    def register(cls, manager: ModelManager):
        """Register a model manager"""
        with cls._lock:
            cls._registry[manager.model_name] = manager
            
            # Start cleanup thread if not already running
            if not cls._running:
                cls._start_cleanup_thread()
    
    @classmethod
    def unregister(cls, model_name: str):
        """Unregister a model manager"""
        with cls._lock:
            if model_name in cls._registry:
                cls._registry[model_name].unload()
                del cls._registry[model_name]
    
    @classmethod
    def get_manager(cls, model_name: str) -> Optional[ModelManager]:
        """Get a model manager by name"""
        with cls._lock:
            return cls._registry.get(model_name)
    
    @classmethod
    def _start_cleanup_thread(cls):
        """Start the background thread for cleaning up idle models"""
        if cls._cleanup_thread and cls._cleanup_thread.is_alive():
            return
        
        cls._running = True
        cls._cleanup_thread = threading.Thread(
            target=cls._cleanup_loop, 
            daemon=True,
            name="model-cleanup-thread"
        )
        cls._cleanup_thread.start()
    
    @classmethod
    def _cleanup_loop(cls):
        """Background thread that periodically checks for and unloads idle models"""
        try:
            logger.info("Starting model cleanup thread")
            
            while cls._running:
                # Sleep first to allow initial models to be used
                time.sleep(60)  # Check every minute
                
                # Check and unload idle models
                cls._cleanup_idle_models()
        except Exception as e:
            logger.error(f"Error in model cleanup thread: {e}")
        finally:
            logger.info("Model cleanup thread stopped")
    
    @classmethod
    def _cleanup_idle_models(cls):
        """Unload any models that have been idle for too long"""
        to_unload = []
        
        # Find models to unload (with lock)
        with cls._lock:
            for model_name, manager in cls._registry.items():
                if manager.is_loaded and manager.should_unload():
                    to_unload.append(model_name)
        
        # Unload models (without holding main registry lock during unloading)
        for model_name in to_unload:
            try:
                manager = cls.get_manager(model_name)
                if manager:
                    logger.info(f"Unloading idle model {model_name} (idle for {manager.idle_time:.1f}s)")
                    manager.unload()
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {e}")
    
    @classmethod
    def stop(cls):
        """Stop the cleanup thread and unload all models"""
        with cls._lock:
            cls._running = False
            
            # Unload all models
            for model_name, manager in list(cls._registry.items()):
                try:
                    manager.unload()
                except Exception as e:
                    logger.error(f"Error unloading model {model_name}: {e}")
            
            # Clear registry
            cls._registry.clear()
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get statistics about all registered models"""
        with cls._lock:
            return {
                "total_models": len(cls._registry),
                "loaded_models": sum(1 for m in cls._registry.values() if m.is_loaded),
                "models": [m.get_stats() for m in cls._registry.values()]
            }


# Create a context manager for using models
def with_model(manager: ModelManager[T]) -> T:
    """
    Context manager for safely using a model.
    
    Example:
        with with_model(my_model_manager) as model:
            result = model.predict(data)
    """
    class ModelContext:
        def __enter__(self) -> T:
            self.model = manager.get_model()
            return self.model
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # No need to do anything on exit - model will be unloaded by cleanup thread if idle
            pass
    
    return ModelContext()