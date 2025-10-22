from agents import Agent, Runner,function_tool
from dotenv import load_dotenv
load_dotenv()

@function_tool
def get_weather(city: str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is sunny"


agent = Agent(name="Assistant", 
             instructions="Always respond in haiku form",
             model="gpt-5-mini",
             tools=[get_weather])
prompt="What is the weather in Tokyo?"


result = Runner.run_sync(agent,prompt)
print(result.final_output)

