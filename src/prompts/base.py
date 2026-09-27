import os
import re

class BaseCodeGen:
    def __init__(self, cfg):
        pass

    def get_completion(self, ):
        raise NotImplementedError
    
    def generate_functions(self,):
        raise NotImplementedError

    def save_code_to_file(self, code, name, save_dir):
        """
        Saves the generated code to a Python file.

        Parameters:
            code (str): The Python code to save.
            filename (str): The filename to save the code to.
        """
        filename_dir = os.path.join(self.prompt_dir, "gen_code", self.cfg.env.name, save_dir)
        if not os.path.exists(filename_dir):
            os.makedirs(filename_dir)
        file_num = len(os.listdir(filename_dir))
        self.gen_code_dir = filename_dir
        prefix, suffix = name.split(".")[0], name.split(".")[1]
        filename = f"{prefix}_{file_num}.{suffix}"
        filename_path = os.path.join(filename_dir, filename)
        with open(filename_path, 'w') as file:
            file.write(code)
        print(f"Code saved to {filename_path}")

    def extract_python_functions(self, text):
        """
        Extracts Python code from an LLM response.

        The LLM is asked to use <code>...</code>, but some models return an
        unclosed <code> block, a markdown block without a language tag, or raw
        Python starting at an import/def line. Accept those forms so valid code
        is not discarded just because the wrapper format is imperfect.
        """
        if text is None:
            raise ValueError("No Python code found in the text")
        text = str(text)
        patterns = [
            r'```python\s*(.*?)```',
            r'```\s*(.*?)```',
            r'<code>\s*(.*?)\s*</code>',
        ]
        for pattern in patterns:
            code = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if code:
                # Some models place the required functions in separate code blocks.
                # Preserve every block in order instead of silently discarding later ones.
                return "\n\n".join(block.strip() for block in code if block.strip())

        code_start = re.search(r'<code>\s*', text, re.IGNORECASE)
        if code_start:
            return text[code_start.end():].strip()

        raw_start = re.search(r'(?m)^(import\s+|from\s+|KNOWLEDGE_ACTIONS\s*=|def\s+)', text)
        if raw_start:
            return text[raw_start.start():].strip()

        raise ValueError("No Python code found in the text")
    

