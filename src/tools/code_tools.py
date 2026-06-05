"""
Code and file manipulation tools for agents.
These tools are bound to agents and called within the graph execution.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> str:
    """
    Read entire contents of a file.
    
    Args:
        file_path: Path to file (absolute or relative to current working directory)
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IsADirectoryError: If path is a directory
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"❌ File not found: {file_path}"
    except IsADirectoryError:
        return f"❌ Path is a directory, not a file: {file_path}"
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file. Creates file if it doesn't exist, overwrites if it does.
    
    Args:
        file_path: Path to file
        content: Content to write
        
    Returns:
        Success message with file path
    """
    try:
        # Create parent directories if they don't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Count lines for feedback
        num_lines = len(content.splitlines())
        return f"✓ Written {num_lines} lines to {file_path}"
    except Exception as e:
        return f"❌ Error writing file: {str(e)}"


@tool
def append_file(file_path: str, content: str) -> str:
    """
    Append content to existing file.
    
    Args:
        file_path: Path to file
        content: Content to append
        
    Returns:
        Success message
    """
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"✓ Appended content to {file_path}"
    except Exception as e:
        return f"❌ Error appending to file: {str(e)}"


@tool
def list_directory(directory_path: str = ".") -> str:
    """
    List files and directories in a folder.
    
    Args:
        directory_path: Path to directory (default: current directory)
        
    Returns:
        Formatted listing of directory contents
    """
    try:
        path = Path(directory_path)
        if not path.is_dir():
            return f"❌ Not a directory: {directory_path}"
        
        items = sorted(path.iterdir())
        if not items:
            return f"(empty directory)"
        
        output = []
        for item in items:
            if item.is_dir():
                output.append(f"📁 {item.name}/")
            else:
                size = item.stat().st_size
                output.append(f"📄 {item.name} ({size} bytes)")
        
        return "\n".join(output)
    except Exception as e:
        return f"❌ Error listing directory: {str(e)}"


@tool
def search_codebase(pattern: str, directory: str = ".", file_extension: str = "*") -> str:
    """
    Search for pattern in files (grep-like behavior).
    
    Args:
        pattern: Text pattern to search for
        directory: Directory to search in (default: current)
        file_extension: File extension filter (e.g., "*.py", "*") 
        
    Returns:
        List of matching lines with file paths
    """
    try:
        path = Path(directory)
        if not path.is_dir():
            return f"❌ Not a directory: {directory}"
        
        matches = []
        pattern_lower = pattern.lower()
        
        for file_path in path.rglob(file_extension):
            if file_path.is_file():
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern_lower in line.lower():
                                matches.append(f"{file_path}:{line_num}: {line.rstrip()}")
                except:
                    pass
        
        if not matches:
            return f"(no matches for '{pattern}')"
        
        return "\n".join(matches[:50])  # Limit to first 50 matches
    except Exception as e:
        return f"❌ Error searching: {str(e)}"


@tool
def run_python_code(code: str) -> str:
    """
    Execute Python code and return output.
    
    ⚠️  Use with caution - executes arbitrary code!
    
    Args:
        code: Python code to execute
        
    Returns:
        stdout and stderr output
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "❌ Code execution timed out (>10s)"
    except Exception as e:
        return f"❌ Error executing code: {str(e)}"


@tool
def create_directory(directory_path: str) -> str:
    """
    Create a directory (creates parent directories if needed).
    
    Args:
        directory_path: Path to create
        
    Returns:
        Success message
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return f"✓ Directory created: {directory_path}"
    except Exception as e:
        return f"❌ Error creating directory: {str(e)}"


@tool
def delete_file(file_path: str) -> str:
    """
    Delete a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Success message
    """
    try:
        Path(file_path).unlink()
        return f"✓ File deleted: {file_path}"
    except FileNotFoundError:
        return f"❌ File not found: {file_path}"
    except Exception as e:
        return f"❌ Error deleting file: {str(e)}"


# All tools available for binding to agents
ALL_TOOLS = [
    read_file,
    write_file,
    append_file,
    list_directory,
    search_codebase,
    run_python_code,
    create_directory,
    delete_file,
]

# Tool subsets for different agents
CODE_EDITING_TOOLS = [
    read_file,
    write_file,
    append_file,
    list_directory,
    search_codebase,
    run_python_code,
    create_directory,
    delete_file,
]

READ_ONLY_TOOLS = [
    read_file,
    list_directory,
    search_codebase,
]
