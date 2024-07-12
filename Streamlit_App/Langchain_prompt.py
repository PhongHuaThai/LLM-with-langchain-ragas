from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
  "Write a delicious recipe for {dish} with a {flavor} twist."
)

# Formatting the prompt with new content
formatted_prompt = prompt_template.format(dish="pasta", flavor="spicy")

print(formatted_prompt)