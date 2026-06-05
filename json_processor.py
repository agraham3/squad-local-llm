import json
from datetime import datetime
from typing import List, Dict, Any, Union

def process_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSON file, filters items where count > 5, sorts by date descending,
    and returns formatted results.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[Dict[str, Any]]: Processed and sorted data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        KeyError: If required fields are missing
        ValueError: If date format is invalid
    """
    try:
        # Read JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Validate data structure
        if not isinstance(data, list):
            raise ValueError("JSON data must be a list of items")
        
        # Filter items where count > 5
        filtered_data = [item for item in data if item.get('count', 0) > 5]
        
        # Sort by date descending
        # Assuming date is in ISO format (YYYY-MM-DD)
        sorted_data = sorted(
            filtered_data,
            key=lambda x: datetime.fromisoformat(x['date']),
            reverse=True
        )
        
        # Format results
        formatted_results = []
        for item in sorted_data:
            formatted_item = {
                'id': item.get('id'),
                'name': item.get('name'),
                'count': item.get('count'),
                'date': item.get('date')
            }
            formatted_results.append(formatted_item)
        
        return formatted_results
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file: {file_path}", e.doc, e.pos)
    except KeyError as e:
        raise KeyError(f"Required field missing in JSON data: {e}")
    except ValueError as e:
        raise ValueError(f"Error processing date: {e}")

# Example usage function (not part of the main function)
def example_usage():
    """Example of how to use the process_json_data function"""
    try:
        # This would be called with an actual file path
        # results = process_json_data('data.json')
        # print(results)
        pass
    except Exception as e:
        print(f"Error: {e}")