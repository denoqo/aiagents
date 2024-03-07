from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

load_dotenv()

population_path = os.path.join('data', 'population.csv')
population_df = pd.read_csv(population_path)


population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
    )

population_query_engine.update_prompts({"pandas_prompt": new_prompt})
population_query_engine.query("What is the population of Canada?")


tool_engines = [
    note_engine,
    QueryEngineTool(
        population_query_engine,
        metadata= ToolMetadata(
            name="information",  
            description="no information")
    )
]

llm = OpenAI("gpt-3.5-turbo")
agent = ReActAgent.from_tools(
    tools= tool_engines,
    llm=llm,
    verbose=True
)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    results = agent.query(prompt)
    print(results)