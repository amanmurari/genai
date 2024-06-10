
from crewai import Agent ,Task,Crew
from langchain_groq import ChatGroq
from crewai_tools import BrowserbaseLoadTool, tools
import os
os.environ["BROWSERBASE_API_KEY"]="<api ley>"
os.environ["GROQ_API_KEY"]='api key'

tool = BrowserbaseLoadTool()
llm=ChatGroq(model="llama3-70b-8192")
agt = Agent(
  role="Senior Data Analyst",
  goal="to extract and prepare for applying ml algorithm."
  "from the data extract insightful approach to grow our business and atract mor customers",
  backstory="you are a senior Data Analyst at a Google company. you have a 100+ years of experience in data analysis.",
 allow_delegation=True,
  verbose=False
  ,llm=llm
)
ml_eng= Agent(
  role="Senior Data Engineer",
  goal="to exteact a data for data analysis to the given source."
  "convert data into structure table form",
    backstory="you are a senior Machine learning Engineer at a Google company. you have a 100+ years of experience in gathering data from diffrent sources.",
  llm=llm,
  verbose= False, allow_delgation=True
)
data_s= Agent(
  role="Senior Data Scientist",
  goal="to chose best machine learning algorithm which has good accuracy, low error or different - different matrics, low erroe for given dataset ."
  "give machine learning algorithm with hyper parameters",
   backstory="you are a senior Data Scientist at a Google company. you have a 100+ years of experience in data Science."
  "you chose best machine learning algorithm",
  llm=llm,
  verbose=False,allow_delegation=False
)

ml_task = Task(
  description='Gather a data from given url {source} and structure data for data analysis',
  expected_output='Data in structure tabular form ',
  tools=[tool],
  agent=ml_eng,
)
da_task = Task(
  description='Analysis the give data and prepare for applying ml algorithm and give insightful decision ',
  expected_output='A bullet list inside decision of the important for business growing and customers attraction',
  agent=agt,
)
ds_task = Task(
  description=' chose best machine learning algorithm',

  expected_output='A bullet list machine learning algorithm of the top 1 most better which has good accuracy',
  agent=data_s,
)

crew = Crew(
    agents=[ml_eng,agt, data_s],
    tasks=[ml_task,da_task,ds_task])
result = crew.kickoff({"source":"https://github.com/datasets/covid-19/blob/main/data/key-countries-pivoted.csv"})
print(result)