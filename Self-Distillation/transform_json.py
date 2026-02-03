#!/usr/bin/env python3
"""
Script to transform JSON from old format to new format.
Expands each item in the old JSON into multiple items in the new JSON,
one for each instruction/golden_answer pair.
"""

import json
import sys


def generate_prompt(name, description, function_descriptions, nl_documentation, instruction):
    """
    Generate the prompt field by concatenating:
    1. Task description
    2. Tool information (name, description, documentation)
    3. Format instructions
    4. The instruction question
    """
    parts = []
    
    # Part 1: Task description
    parts.append("Your task is to answer the user's question using available tools.")
    
    # Part 2: Tool information
    tool_info = f"\nYou have access to the following tools:\nName: {name}\nDescription: {description}\nDocumentation:\n{nl_documentation}"
    parts.append(tool_info)
    
    # Part 3: Format instructions
    format_instructions = (
        "\nUse the following format:\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of the tool names.\n"
        "Action Input: the input to the action, must be in JSON format. "
        "All of the action input must be realistic and from the user.\n\n"
        "Begin!"
    )
    parts.append(format_instructions)
    
    # Part 4: The instruction question
    parts.append(f"Question: {instruction}")
    
    return " ".join(parts)


def transform_json(old_data):
    """
    Transform old JSON format to new JSON format.
    Each item in old format gets expanded into multiple items in new format,
    one for each instruction/golden_answer pair.
    """
    new_data = []
    
    for item in old_data:
        # Extract fields from old format
        name = item.get("Name", "")
        description = item.get("Description", "")
        category = item.get("Category", "")
        nl_documentation = item.get("NLDocumentation", "")
        function_descriptions = item.get("Function_Description", {})
        instructions = item.get("Instructions", [])
        golden_answers = item.get("Golden_Answers", [])
        
        # Create one new item for each instruction/golden_answer pair
        for i, (instruction, golden_answer) in enumerate(zip(instructions, golden_answers)):
            # Generate the prompt
            prompt = generate_prompt(
                name, 
                description, 
                function_descriptions,
                nl_documentation,
                instruction
            )
            
            # Create new item
            new_item = {
                "prompt": prompt,
                "name": name,
                "description": description,
                "nl_documentation": nl_documentation,
                "instruction": instruction,
                "golden_answer": golden_answer
            }
            
            new_data.append(new_item)
    
    return new_data


def main():
    """Main function to read, transform, and write JSON."""
    if len(sys.argv) < 3:
        print("Usage: python transform_json.py <input_file> <output_file>")
        print("Example: python transform_json.py old_format.json new_format.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Read old JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        
        # Transform to new format
        new_data = transform_json(old_data)
        
        # Write new JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully transformed {len(old_data)} old items into {len(new_data)} new items")
        print(f"Output written to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()