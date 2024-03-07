from llama_index.core.tools import FunctionTool
import os

note_file = os.path.join('data', 'note.txt')

def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, 'w')

    with open(note_file, 'a') as f:
        f.writelines([note + '\n'])

    return "note saved"


note_engine = FunctionTool.from_defaults(
    save_note, 
    "note_saver", 
    "Save a note to a file"
)