import os

from langchain_core.messages import HumanMessage, SystemMessage
import io
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
    create_csv_agent,
)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm_name = "gpt-3.5-turbo"

gpt_model = ChatOpenAI(api_key=openai_api_key, name=llm_name)

df = pd.read_csv("./data/salaries_2023.csv")
df.fillna(0, inplace=True)

agent = create_pandas_dataframe_agent(
    llm=gpt_model,
    df=df,
    verbose=True,
    allow_dangerous_code=True,
)


answer = agent.invoke("How many rows are there in the dataframe?")
print(answer)