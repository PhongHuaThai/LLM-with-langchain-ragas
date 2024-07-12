import json
import re

class Handle_question_list:
  json_file_path = 'data/question_list.json'

  def load_json(self):
    with open(self.json_file_path, 'r') as file:
        return json.load(file)
    
  def save_json(self, data):
    with open(self.json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

  def extract_question(self, prompt_template):
      question_match = re.search(r'<<QUESTION>>(.*?)<</QUESTION>>', prompt_template)
      if question_match:
        question = question_match.group(1).strip()

        result_string = re.sub(r'<<QUESTION>>.*?<</QUESTION>>', r'{input}', prompt_template)
        
        return question, result_string
      else:
        return None, prompt_template