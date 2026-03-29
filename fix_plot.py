import re

with open("plot_saved_forward_data.py", "r", encoding="utf-8") as f:
    content = f.read()

parts = content.split("if __name__ == '__main__':")
if len(parts) == 2:
    main_part = "if __name__ == '__main__':" + parts[1].split("def get_mt_freqs():")[0]
    func_part = "def get_mt_freqs():" + parts[1].split("def get_mt_freqs():")[1]
    
    new_content = parts[0] + "\n" + func_part + "\n" + main_part
    with open("plot_saved_forward_data.py", "w", encoding="utf-8") as f:
        f.write(new_content)
