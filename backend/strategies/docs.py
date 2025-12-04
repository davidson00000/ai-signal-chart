import yaml
import os
from typing import Tuple, Dict, Any, Optional

def parse_strategy_doc(file_path: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Parse a markdown file with YAML frontmatter.
    
    Args:
        file_path: Path to the markdown file.
        
    Returns:
        Tuple containing:
        - Markdown content (str) without frontmatter.
        - Presets dictionary (Dict) parsed from frontmatter, or None if not found.
    """
    if not os.path.exists(file_path):
        return "# Document not found", None
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check for YAML frontmatter
    if content.startswith("---\n"):
        try:
            parts = content.split("---\n", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1]
                markdown_content = parts[2]
                
                # Parse YAML
                frontmatter = yaml.safe_load(frontmatter_str)
                presets = frontmatter.get("presets")
                
                return markdown_content, presets
        except Exception as e:
            print(f"Error parsing frontmatter in {file_path}: {e}")
            # Fallback to returning raw content if parsing fails
            return content, None
            
    return content, None
