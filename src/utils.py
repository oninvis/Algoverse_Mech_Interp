from pathlib import Path
from datetime import datetime

def get_repo_root() -> Path:
    """Obtains the path to the root of the github repository.
    
    This is done by iteratively going through parents until a directory with `.git` is found.
    
    Returns:
        pathlib.Path: The path to the repository root.
        
    Raises:
        FileNotFoundError: If no .git directory is found.
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / '.git').is_dir():
            return parent
    raise FileNotFoundError("Could not find git repository root.")

def get_current_time_str() -> str:
    """Returns the current time in YYYYMMDD-HHMMSS format.
    
    Returns:
        str: Current time formatted as YYYYMMDD-HHMMSS
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")

