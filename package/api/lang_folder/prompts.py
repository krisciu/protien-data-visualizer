from lang_folder.templates import INPUT_CLASSIFICATION_PROMPT_TEMPLATE, FORMAT_ANSWER_FROM_QUERY_TEMPLATE, SYSTEM_PROMPT_FOR_QUERY_GENERATION_TEMPLATE, SYSTEM_PROMPT_FOR_NORMAL_CONVERSATION_TEMPLATE, SYSTEM_PROMPT_FOR_FOLLOW_UP_QUESTION_GENERATION, TABLE_DETAILS_PROMPT_TEMPLATE, CHART_CLASSIFICATION_PROMPT_TEMPLATE, SYSTEM_PROMPT_FOR_CHART_GENERATION_TEMPLATE, SYSTEM_PROMPT_FOR_CHART_CONVERSATION_TEMPLATE
from langchain_openai import OpenAIEmbeddings
from lang_folder.few_shot_examples import few_shot_examples
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from lang_folder.vectorStore.pineconeClient import PineconeClient


INPUT_CLASSIFICATION_PROMPT = PromptTemplate.from_template(
    INPUT_CLASSIFICATION_PROMPT_TEMPLATE
)

CHART_CLASSIFICATION_PROMPT = PromptTemplate.from_template(
    CHART_CLASSIFICATION_PROMPT_TEMPLATE
)

ANSWER_USER_QUESTION_PROMPT = PromptTemplate.from_template(
    FORMAT_ANSWER_FROM_QUERY_TEMPLATE
)

TABLE_DETAILS_PROMPT  = PromptTemplate.from_template(
    TABLE_DETAILS_PROMPT_TEMPLATE
)


# Prompt to generate chart schema based on table information and current conversation
CHART_SPEC_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_FOR_CHART_GENERATION_TEMPLATE),
    ("placeholder", "{conversation}"),
    ("human", "{question}\n\nSPEC: "),
])

# Conversations about generating charts
GENERATE_CHART_CONVERSATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_FOR_CHART_CONVERSATION_TEMPLATE),
    # Means the template will receive an optional list of messages under
    # the "conversation" key
    ("placeholder", "{conversation}")
    # Equivalently:
    # MessagesPlaceholder(variable_name="conversation", optional=True)
])

# Normal Conversation with user
GENERATE_CONVERSATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_FOR_NORMAL_CONVERSATION_TEMPLATE),
    # Means the template will receive an optional list of messages under
    # the "conversation" key
    ("placeholder", "{conversation}")
    # Equivalently:
    # MessagesPlaceholder(variable_name="conversation", optional=True)
])

# Follow up questions that user can ask
GENERATE_FOLLOW_UP_CONVERSATION_PROMPT = ChatPromptTemplate.from_messages([
   ("system", SYSTEM_PROMPT_FOR_FOLLOW_UP_QUESTION_GENERATION),
   ("placeholder", "{conversation}")
])

# Few Shot Examples for forming the queries
# The passed in examples should be of type 
# {"input": "String", "query": "SQL Query String"}
_example_prompt_for_few_shot = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

# This few shot prompt is just like we pass a few examples to 
# the LLM as support before we let it answer any question
FEW_SHOT_PROMPT = FewShotChatMessagePromptTemplate(
    example_prompt=_example_prompt_for_few_shot,
    examples=few_shot_examples, # {"input": "String", "query": "SQL Query String"}[]
    input_variables=["input"], # The variable holds the value sent from the user 
)

def _getSemanticExampleSelectorChain(top_k=3, input_field="input"):
    # Initialize the PineconeClient
    #TODO: fix structure so we can use dependency injection as in main.py
    pinecone_client = PineconeClient()
    pinecone_client.load_few_shot_examples(few_shot_examples)
    
    # Create the SemanticSimilarityExampleSelector using the PineconeClient
    return SemanticSimilarityExampleSelector(
        vectorstore=pinecone_client,  
        k=top_k,  
        input_keys=[input_field] 
    )

# Whenever this prompt is used, it comes along with the few shot examples selected
DYNAMIC_FEW_SHOT_PROMPT_WITH_EXAMPLE_SELECTION = FewShotChatMessagePromptTemplate(
    example_prompt=_example_prompt_for_few_shot,
    example_selector=_getSemanticExampleSelectorChain(), # Here rather than passing in all the examples, we pass in the example selector object 
    input_variables=["input","top_k", "table_info"],
)


# Combined prompt that takes in the following
# 1. System prompt
# 2. Few shot examples
# 3. User query
# With this prompt, we aim to pass in examples for the llm to generate a query
# This function needs an input parameter in the field "input" --> this is the user query
GENERATE_QUERY_PROMPT_WITH_FEW_SHOT_SELECTION = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_FOR_QUERY_GENERATION_TEMPLATE),
        DYNAMIC_FEW_SHOT_PROMPT_WITH_EXAMPLE_SELECTION,
        ("human", "{input}\n\nSQL: "),
    ]
)
