""" Decides which analysis path a loaded file should take based on its type. """

from core.loaders import LoadedFile

MIN_TEXT_CHARACTERS = 150 # less text then this is probably not worth analyzing

def choose_path(loaded_file: LoadedFile) -> str:
    
    """ Return 'tabular', 'text', 'both' or 'empty'."""
    has_tables = len(loaded_file.tables) >0
    has_text = len(loaded_file.text) >= MIN_TEXT_CHARACTERS
    
    if has_tables and has_text:
        return "both"
    elif has_tables:
        return "tabular"
    elif has_text:
        return "text"
    return "empty"

