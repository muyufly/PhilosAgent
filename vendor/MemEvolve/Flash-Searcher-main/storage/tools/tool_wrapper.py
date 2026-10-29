"""
Unified Tool Wrapper for Memory Systems

This module provides utilities to wrap Python functions as FlashOAgents Tool objects.
Memory providers can use this to convert stored Python functions into executable tools.

Usage:
    from storage.tools.tool_wrapper import ToolWrapper
    
    wrapper = ToolWrapper(model=your_model, logger=your_logger)
    wrapped_tool = wrapper.wrap_function(func, func_name)
"""

import inspect
import logging
from typing import Any, Callable, Optional


class ToolWrapper:
    """
    Wraps Python functions as FlashOAgents Tool objects
    
    This class handles the conversion of raw Python functions into Tool objects
    that can be injected into agents. It supports:
    - Automatic parameter type inference
    - Model injection for LLM-powered tools
    - Parameter-less functions
    - Type hint handling (int, float, str, bool, dict, list)
    - Error handling during tool execution
    """
    
    def __init__(self, model: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the tool wrapper
        
        Args:
            model: Optional LLM model to inject into tools that need it
            logger: Optional logger for debugging
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self._cache = {}  # Cache wrapped tools to avoid re-wrapping
    
    def wrap_function(self, func: Callable, func_name: str) -> Optional[Any]:
        """
        Wrap a Python function as a FlashOAgents Tool object
        
        Args:
            func: Python function to wrap
            func_name: Name for the tool
            
        Returns:
            Wrapped Tool object, or None if wrapping failed
        """
        # Check cache
        if func_name in self._cache:
            return self._cache[func_name]
        
        try:
            from FlashOAgents.tools import Tool
            
            # Validate input
            if not inspect.isfunction(func):
                self.logger.warning(f"'{func_name}' is not a function, cannot wrap as tool")
                return None
            
            # Extract function metadata
            sig = inspect.signature(func)
            docstring = inspect.getdoc(func) or "No description available"
            func_params = list(sig.parameters.keys())
            needs_model = 'model' in func_params
            
            # Build tool inputs schema
            tool_inputs = self._build_tool_inputs(sig)
            
            # Handle parameter-less functions
            has_no_params = len(tool_inputs) == 0
            if has_no_params:
                tool_inputs = {"query": {"type": "string", "description": "This parameter is not used"}}
            
            # Generate forward method dynamically
            forward_code = self._generate_forward_method(
                tool_inputs=tool_inputs,
                has_no_params=has_no_params,
                needs_model=needs_model
            )
            
            # Create Tool class dynamically
            wrapped = self._create_dynamic_tool(
                tool_name=func_name,
                tool_func=func,
                docstring=docstring,
                tool_inputs=tool_inputs,
                forward_code=forward_code
            )
            
            # Cache and return
            self._cache[func_name] = wrapped
            self.logger.info(f"Tool '{func_name}' wrapped (model={'injected' if needs_model else 'not needed'})")
            return wrapped
            
        except Exception as e:
            self.logger.error(f"Tool wrapping failed for '{func_name}': {str(e)}", exc_info=True)
            return None
    
    def _build_tool_inputs(self, sig: inspect.Signature) -> dict:
        """
        Build tool inputs schema from function signature
        
        Args:
            sig: Function signature
            
        Returns:
            Dictionary mapping parameter names to type schemas
        """
        tool_inputs = {}
        
        for param_name, param in sig.parameters.items():
            # Skip special parameters
            if param_name in ("self", "model"):
                continue
            
            # Infer parameter type
            param_type = self._infer_param_type(param.annotation)
            
            tool_inputs[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
        
        return tool_inputs
    
    def _infer_param_type(self, annotation: Any) -> str:
        """
        Infer parameter type from annotation
        
        Args:
            annotation: Parameter annotation
            
        Returns:
            Type string (integer, number, boolean, string, dict, array, any)
        """
        # Default type
        if annotation == inspect.Parameter.empty:
            return "string"
        
        # Direct type mapping
        type_map = {
            int: "integer",
            float: "number",
            bool: "boolean",
            str: "string",
            dict: "dict",
            list: "array",
        }
        
        # Check direct type
        if annotation in type_map:
            return type_map[annotation]
        
        # Handle string type hints (for generated code)
        if isinstance(annotation, str):
            annotation_lower = annotation.lower()
            if "int" in annotation_lower:
                return "integer"
            elif "float" in annotation_lower or "number" in annotation_lower:
                return "number"
            elif "bool" in annotation_lower:
                return "boolean"
            elif "str" in annotation_lower or "string" in annotation_lower:
                return "string"
            elif "dict" in annotation_lower:
                return "dict"
            elif "list" in annotation_lower or "array" in annotation_lower:
                return "array"
        
        # Fallback to 'any' for complex types
        return "any" if hasattr(annotation, "__origin__") else "any"
    
    def _generate_forward_method(
        self, 
        tool_inputs: dict, 
        has_no_params: bool, 
        needs_model: bool
    ) -> str:
        """
        Generate forward method code dynamically
        
        Args:
            tool_inputs: Tool input schema
            has_no_params: Whether function has no parameters
            needs_model: Whether function needs model injection
            
        Returns:
            Forward method code as string
        """
        params_str = ", ".join(tool_inputs.keys())
        
        if has_no_params:
            if needs_model:
                return f"""
def forward(self, {params_str}):
    try:
        result = self._func(model=self._model)
        return str(result) if result is not None else ""
    except Exception as e:
        return f"Tool execution error: {{str(e)}}"
"""
            else:
                return f"""
def forward(self, {params_str}):
    try:
        result = self._func()
        return str(result) if result is not None else ""
    except Exception as e:
        return f"Tool execution error: {{str(e)}}"
"""
        else:
            kwargs_str = ", ".join([f"{p}={p}" for p in tool_inputs.keys()])
            if needs_model:
                return f"""
def forward(self, {params_str}):
    try:
        result = self._func({kwargs_str}, model=self._model)
        return str(result) if result is not None else ""
    except Exception as e:
        return f"Tool execution error: {{str(e)}}"
"""
            else:
                return f"""
def forward(self, {params_str}):
    try:
        result = self._func({kwargs_str})
        return str(result) if result is not None else ""
    except Exception as e:
        return f"Tool execution error: {{str(e)}}"
"""
    
    def _create_dynamic_tool(
        self,
        tool_name: str,
        tool_func: Callable,
        docstring: str,
        tool_inputs: dict,
        forward_code: str
    ) -> Any:
        """
        Create a dynamic Tool class and instantiate it
        
        Args:
            tool_name: Name of the tool
            tool_func: Original function
            docstring: Function docstring
            tool_inputs: Tool input schema
            forward_code: Forward method code
            
        Returns:
            Instantiated Tool object
        """
        from FlashOAgents.tools import Tool
        
        # Execute forward method code
        local_namespace = {}
        exec(forward_code, {"str": str}, local_namespace)
        
        # Create dynamic Tool class
        class DynamicTool(Tool):
            name = tool_name
            description = docstring.split('\n')[0] if docstring else tool_name
            inputs = tool_inputs
            output_type = "string"
            
            def __init__(self):
                super().__init__()
                self._func = tool_func
                self._model = None
        
        # Attach forward method
        DynamicTool.forward = local_namespace['forward']
        
        # Instantiate and configure
        wrapped = DynamicTool()
        wrapped._model = self.model
        
        return wrapped
    
    def clear_cache(self):
        """Clear the wrapped tools cache"""
        self._cache.clear()

