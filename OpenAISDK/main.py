from agents import Agent, Runner
from dotenv import load_dotenv
load_dotenv()

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

prompt="Write a haiku about recursion in programming."

result = Runner.run_sync(agent,prompt)
print(result.final_output)

