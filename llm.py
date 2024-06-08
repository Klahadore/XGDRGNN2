import ollama
# response = ollama.chat(model='llama3', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])


def get_text(text):
  response = ollama.chat(model="llama3", messages=[
    {
      'role': 'user',
      'content': f"Provide a comprehensive and detailed definition of the medical term '{text}'. Focus solely on explaining what the term means."
    },
  ])
  return response['message']['content']







if __name__ == "__main__":
  print(get_text("breast cancer"))



