import dspy
import os
from dotenv import load_dotenv

load_dotenv()

# model_name=os.getenv('MODEL_NAME')
# api_key=os.getenv('API_KEY')
# api_base=os.getenv('API_BASE')
#
dspy.configure(lm=dspy.LM('ollama_chat/chevalblanc/gpt-4o-mini'))
# Define a calculator tool
def calculator(expression: str) -> float:
    print("tool was called.....")
    return eval(expression)

# Create the agent module
class CalculatorAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # Register the calculator tool with the ReAct module
        self.react = dspy.ReAct("question -> answer", tools=[calculator])

    def forward(self, question: str):
        return self.react(question=question)

agent = CalculatorAgent()
print(agent("What is 2 + 2 * 5?"))
# # lm = dspy.LM(model_name,api_key=api_key,api_base=api_base)
# #
# # dspy.configure(lm=lm)
#
#
# class MathQA(dspy.Module):
#     def __init__(self):
#         super().__init__()
#         # Define the module using Chain‑of‑Thought reasoning
#         self.solve = dspy.ChainOfThought("question -> answer: float")
#
#     def forward(self, question: str):
#         return self.solve(question=question)
#
# # Instantiate and invoke the module
# qa = MathQA()
# result = qa("What is 3 * 7 + 2?")
# print(result)

class Summarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("document -> summary")

    def forward(self, document: str):
        return self.summarize(document=document)

# Example
doc = "DSPy is a framework for programming language models..."
summary = Summarizer()(doc)
print(summary)

